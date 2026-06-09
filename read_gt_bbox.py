import numpy as np

# box_path = 'mini_dataset/train\moorea_2019-06-21_000_854500000_914500000_bbox.npy'
# box_path = 'E:/My_big_data/Prophesee/boom_simulation_5_21/label_creation/1/521_1_bbox.npy'
box_path = '2025-7-8_preprocessed/test/20-57-40_2_bbox.npy'
box_events = np.load(box_path)
print(box_events)
# print(box_events[:])
print(box_events.dtype)
print(len(box_events))

'''
bbox.dtype:

[('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')]

'''
