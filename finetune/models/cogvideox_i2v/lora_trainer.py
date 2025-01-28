from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register

import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
from ..train_utils.unimatch.unimatch.unimatch import UniMatch
from ..train_utils.unimatch.utils.flow_viz import flow_to_image


FLOW_SCALE = 128
PROMPT = "a realistic driving scenario with high visual quality, the overall scene is moving forward."


palette = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)  # For tracking IDs


def preprocess_size(image1, image2, padding_factor=32):
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    # inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                 int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    inference_size = [384, 512]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
    
    return image1, image2, inference_size, ori_size, transpose_img

def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr

@torch.no_grad()
def get_optical_flow(unimatch, video_frame):
    '''
        video_frame: [b, t, c, w, h]
    '''
    video_dtype = video_frame.dtype
    flows = []
    for i in range(video_frame.shape[1] - 1):
        image1, image2 = video_frame[:, i], video_frame[:, i+1]

        image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
        results_r = unimatch(image1_r, image2_r,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
            )['flow_preds'][-1]
        flows.append(postprocess_size(results_r, inference_size, ori_size, transpose_img).unsqueeze(1)) 
    
    return torch.cat(flows, dim=1).to(video_dtype)

@torch.no_grad()
def get_seg_map(yolo_model, video_frame):
    frame_height, frame_width = video_frame.shape[2:]
    mask_images = []
    mask0 = None
    if yolo_model.predictor is not None:
        for tracker in yolo_model.predictor.trackers:
            tracker.reset()
            tracker.reset_id()

    # for frame_id, result in enumerate(tracking_results[:-1]):
    for frame_id, frame in enumerate(video_frame):
        result = yolo_model.track(
            source=frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 
            persist=True, verbose=False, conf=0.2, iou=0.2, 
            tracker="bytetrack.yaml")[0]
        
        # Create empty images for masks
        mask_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Extract masks and detection information
        masks = result.masks
        boxes = result.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else []

        if masks is not None:
            if frame_id == 0:
                mask0 = masks.data

            for i, mask in enumerate(masks.data):
                # Convert mask to a binary mask
                binary_mask = mask.cpu().numpy().astype(np.uint8)
                binary_mask = cv2.resize(binary_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

                track_id = int(ids[i]) if len(ids) > i else 0
                color = palette[track_id % len(palette)].tolist()
                colored_mask = np.zeros_like(mask_image)
                for c in range(3):
                    colored_mask[:, :, c] = binary_mask * color[c]

                mask_image = cv2.addWeighted(mask_image, 1, colored_mask, 0.5, 0)

        mask_images.append(torch.from_numpy(mask_image).permute(2, 0, 1).unsqueeze(0))

    return torch.cat(mask_images, dim=0).unsqueeze(0), mask0

@torch.no_grad()
def get_seg_flow(yolo_model, unimatch, video_frames, perterb=True):
    '''
        Input:
            video_frames: [b, t, c, h, w]
        Output (fp16, cuda):
            optical_flow: [b, t-1, 2, h, w], 
            seg_map: [b, t-1, 3, h, w].
    '''
    video_frames = (video_frames + 1) * 127.5
    flow = get_optical_flow(unimatch, video_frames)  # flow, direct_flow: [b, f-1, 2, h, w]
    flow = torch.cat([flow, torch.zeros_like(flow[:, 0:1])], dim=1)

    mask_image_batch = []
    for b, video_frame in enumerate(video_frames):   # i: batch, j: frame
        s_maps, _ = get_seg_map(yolo_model, video_frame)
        mask_image_batch.append(s_maps.to(video_frames.device))       

    mask_image_batch = torch.cat(mask_image_batch, dim=0).to(torch.float16)  # torch.Size([1, 24, 3, 256, 512]), [0, 1], fp16
    
    return flow, mask_image_batch


class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        components.flow_model = UniMatch(feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow')
        checkpoint = torch.load('/hpc2hdd/home/txu647/code/CogVideo/finetune/models/train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
        components.flow_model.load_state_dict(checkpoint['model'])

        components.seg_model = YOLO('/hpc2hdd/home/txu647/code/segment-anything-2/yolo11l-seg.pt')  # Replace with your model variant if needed

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], [-1, 1]
        vae = self.components.vae
        seg_model = self.components.seg_model
        flow_model = self.components.flow_model
        video = video.to(vae.device)
        
        flow, seg = get_seg_flow(seg_model, flow_model, video.permute(0, 2, 1, 3, 4))

        seg = seg.to(vae.dtype).contiguous()
        flow = flow.to(vae.dtype).contiguous()
        video = video.to(vae.dtype)

        latent = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor

        return latent, seg, flow

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [],
               "video_seg": [], "video_flow": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            video_seg = sample["video_seg"]
            video_flow = sample["video_flow"]

            flow_images = []
            for flow in video_flow:
                flow_image = flow_to_image(flow.permute(1, 2, 0).float(), maxrad=FLOW_SCALE)
                flow_images.append(torch.from_numpy(flow_image))
            flow_images = torch.stack(flow_images).permute(3, 0, 1, 2) / 127.5 - 1.0

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["video_seg"].append(video_seg)
            ret["video_flow"].append(flow_images)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["video_seg"] = torch.stack(ret["video_seg"])
        ret["video_flow"] = torch.stack(ret["video_flow"])
        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        vae = self.components.vae

        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["images"]
        video_seg = batch["video_seg"].permute(0, 2, 1, 3, 4)
        video_flow = batch["video_flow"].to(latent.dtype)
        
        latent_seg = vae.encode(video_seg).latent_dist.sample() * vae.config.scaling_factor
        latent_flow = vae.encode(video_flow).latent_dist.sample() * vae.config.scaling_factor

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        condition_latent = torch.cat([latent_seg, latent_flow], dim=1).permute(0, 2, 1, 3, 4)
        _latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)
        latent_img_noisy = torch.cat([_latent_img_noisy, condition_latent], dim=1)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames * 2,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]
        predicted_noise = predicted_noise[:, :7]  # skip conditional latents
        
        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        vae = self.components.vae.to(self.accelerator.device)

        prompt = PROMPT
        image = eval_data["images"]
        video_seg = eval_data["video_seg"].permute(0, 2, 1, 3, 4).to(vae.device, dtype=vae.dtype)
        video_flow = eval_data["video_flow"].to(vae.device, dtype=vae.dtype)

        latent_seg = vae.encode(video_seg).latent_dist.sample() * vae.config.scaling_factor
        latent_flow = vae.encode(video_flow).latent_dist.sample() * vae.config.scaling_factor
        condition_latent = torch.cat([latent_seg, latent_flow], dim=1).permute(0, 2, 1, 3, 4)

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            condition_latent=condition_latent,
            generator=self.state.generator,
            num_inference_steps=25,  # tianshuo
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
