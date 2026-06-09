#这是用来查看预处理的h5文件的各个attr

import h5py
import numpy as np

with h5py.File('mini_dataset_processed/train/moorea_2019-06-21_000_854500000_914500000.h5', "r") as f:
    for key, value in f['data'].attrs.items():
        print(f"{key} = {value}")
    
    data=f['data']
    # 将数据转换成 NumPy 数组
    data_np = data[:]  # 等同于 np.array(data)
    
    #一般情况下有正有负。注意：运行速度很慢，因为数据很大。
    print(np.min(data_np))
    print(np.max(data_np))
    
    print("数据形状：", data_np.shape)
    print("数据类型：", data_np.dtype)
    
    # # 打印前两帧的数据（防止输出太多）
    # print("前两帧数据：")
    # print(data_np[:2])  # 注意：这里打印的是前两帧的完整数据，可能还是很大