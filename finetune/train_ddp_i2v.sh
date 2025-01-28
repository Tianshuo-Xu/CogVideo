#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "debug"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/hpc2hdd/home/txu647/code/video_data/clips"
    --caption_column "not_important"
    --video_column "valid_videos.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "25x480x720"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000 # save checkpoint every x steps
    --checkpointing_limit 20 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "NotImportant"
    --validation_steps 1000  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 8
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"

# --config_file 0.yaml
# bash train_ddp_i2v.sh