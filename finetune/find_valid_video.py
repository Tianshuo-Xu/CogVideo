import cv2
import concurrent.futures
from tqdm import tqdm  # 可选，用于显示进度条

def is_video_valid(video_path):
    """检查单个视频是否可读"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, _ = cap.read()
        cap.release()
        return ret, video_path
    except Exception as e:
        return False, video_path

def filter_valid_videos(input_txt, output_txt, max_workers=8):
    # 读取所有视频路径
    with open(input_txt, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]

    # 多线程检查视频有效性
    valid_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = [executor.submit(is_video_valid, path) for path in video_paths]
        # 使用tqdm显示进度（可选）
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            ret, path = future.result()
            if ret:
                valid_paths.append(path)

    # 写入有效路径到文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        for path in valid_paths:
            f.write(path + '\n')

    print(f"有效视频数量: {len(valid_paths)}，已保存到 {output_txt}")

# 示例调用
input_txt = 'video.txt'   # 输入文件路径
output_txt = 'valid_videos.txt' # 输出文件路径
max_workers = 32                 # 根据CPU核心数调整（建议4-16）

filter_valid_videos(input_txt, output_txt, max_workers)