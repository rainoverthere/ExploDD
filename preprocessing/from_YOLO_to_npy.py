#该脚本暂时设定的是逐个场景生成对应的label.npy
#把该场景所包含的所有txt文件放在一个文件夹里进行处理
#frame_extraction.py生成的帧的时间戳实质上是每个ev_repr的右边界，都要减去0.5倍delta_t才真正包含于对应ev_repr (delta_t = 1/fps 而不一定是accumulation_time。accumulation_time可以大于delta_t来过度累计以增强视觉效果方便标注)
#比如在第一次尝试中，导出的视频为1000fps，delta_t=1ms，帧的时间戳依次为1ms,2ms,3ms,...,但每一帧label的时间戳要减去0.5*delta_t=0.5ms才作为最终存进去的bbox时间戳

#注意：line 13 14 32 需要根据实际情况设定

import os
import numpy as np
import re

# # 真实数据
# label_folder = '2025-7-8-labels-creation-new/21-18/4/labels'  # 修改为目标标签文件夹路径 ===============================================================
# out_path='2025-7-8-labels-creation-new/labels_preprocessed/21-18_4_bbox.npy'#===============================================================================

# blender模拟数据
label_folder = 'E:\My_big_data\Prophesee\KHAOS\KHAOS_v2e_out_for_training\seq0089_cut300Hz_p0.4_n0.02\labels_creation\seq0089_slow_far_view3_0004-0055/labels'
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(label_folder))), 'labels_preprocessed')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, os.path.basename(os.path.dirname(label_folder))+'_bbox.npy')

img_width = 1280
img_height = 720

# 定义结构化数据类型
dtype = [('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'),
         ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')]

# 收集所有.txt文件
label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt') and f.startswith('frame_')],
                     key=lambda x: int(re.findall(r'\d+', x)[0]))  # 按时间戳排序 这里的key即为取出(第一个)数字部分然后转化为整数，即时间戳

results = []
track_id = 1  # 初始化track_id

for filename in label_files:
    filepath = os.path.join(label_folder, filename)
    # 提取 微秒 时间戳
    # ts = int(re.findall(r'\d+', filename)[0]) - 100 #用每个ev_repr的中间时刻来作为事件帧(label帧)的时间戳 #================================================================
    ts = int(re.findall(r'\d+', filename)[0]) #不再减去0.5*delta_t
    # ts = int(ms * 1000)


    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f'{filepath}的格式存在问题,某一行的长度不为5!!!')
                continue  # 跳过格式不对的行

            class_id = int(parts[0])
            x_c, y_c, w_norm, h_norm = map(float, parts[1:])

            # 反归一化
            w = w_norm * img_width
            h = h_norm * img_height

            # 左上角坐标（从归一化中心位置转换）并 -1
            x = round(x_c * img_width - w / 2)- 1
            y = round(y_c * img_height - h / 2) - 1

            w = round(w)
            h = round(h)

            # 存储为结构化数组条目
            results.append((ts, x, y, w, h, class_id, 1.0, track_id))
            track_id += 1

# 转换为结构化数组并保存
structured_array = np.array(results, dtype=dtype)
np.save(out_path, structured_array)
print(f"成功保存 {len(structured_array)} 条数据到 {out_path}")
