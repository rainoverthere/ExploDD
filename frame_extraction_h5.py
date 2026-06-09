import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

def load_delta_t(h5path, index):
    with h5py.File(h5path, "r") as f:
        assert f["data"].attrs.get("mode", "delta_t") == "delta_t", "only delta_t mode supported"
        assert f["data"].attrs["delta_t"] > 0
        array = f['data'][index]  # [c,h,w]
        return array

def viz_event_cube_rgb(im):
    summed = np.sum(im, axis=0)
    h, w = summed.shape
    # 设置颜色
    rgb = np.full((h, w, 3), 127, dtype=np.uint8)  # 默认为灰色
    rgb[summed > 0] = [255, 255, 255]  # 白色
    rgb[summed < 0] = [0, 0, 0]  # 黑色
    return rgb

def process_h5_files(root_dir, output_dir="2025-7-8-labels-creation-new"):#====================================================
    # 遍历train/val/test三个子文件夹
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        # 遍历split目录下的所有h5文件
        for h5_file in tqdm(os.listdir(split_dir), desc=f"Processing {split} files"):
            if not h5_file.endswith(".h5"):
                continue
                
            h5_path = os.path.join(split_dir, h5_file)
            
            # 解析文件名 如20-57-40_2.h5
            base_name = os.path.splitext(h5_file)[0]
            time_part, seq_part = base_name.split("_")
            seq_num = seq_part
            
            # 创建输出目录
            frames_dir = os.path.join(output_dir, time_part, seq_num, "frames")
            labels_dir = os.path.join(output_dir, time_part, seq_num, "labels")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # 读取h5文件中的所有delta_t
            with h5py.File(h5_path, "r") as f:
                assert f["data"].attrs.get("mode", "delta_t") == "delta_t", "only delta_t mode supported"
                assert f["data"].attrs["delta_t"] > 0
                num_frames = f["data"].shape[0]
                timestamp_us = 0  # 从0开始
                
                # 处理每个delta_t
                for i in range(num_frames):
                    array = f['data'][i]  # [c,h,w]
                    rgb_frame = viz_event_cube_rgb(array)
                    
                    # 保存RGB帧，使用时间戳命名
                    timestamp_us += 100 # # 时间戳增加100us(10000fps)#=======================================================================
                    frame_filename = f"frame_{timestamp_us:08d}us.png"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, rgb_frame)
                    # print(f"保存: {frame_path}")
                    
            print(f"Processed {h5_file}: saved {num_frames} frames to {frames_dir}")

if __name__ == "__main__":
    root_dir = "2025-7-8_zjy_lzc_events_preprocessed"#===========================================里面需要有train/val/test
    output_dir="2025-7-8-labels-creation-new"#===================================================
    process_h5_files(root_dir=root_dir,output_dir=output_dir)