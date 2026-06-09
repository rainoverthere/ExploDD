import numpy as np

def filter_new_events_with_gaussian(new_events, sxf, syf):
    """
    针对 new_events (N*4 float32: [t,x,y,p]) 做二维高斯筛选（像素级，向量化优化）。
    - t: 秒(float)
    - x,y: 像素坐标
    - p: +1/-1
    """
    if new_events is None or len(new_events) == 0:
        return new_events

    # Step 1: 统计像素正负事件数量
    coords_pos, counts_pos = np.unique(
        new_events[new_events[:,3] == 1][:,1:3].astype(int),
        return_counts=True, axis=0
    )
    coords_neg, counts_neg = np.unique(
        new_events[new_events[:,3] == -1][:,1:3].astype(int),
        return_counts=True, axis=0
    )

    pos_dict = {tuple(c): counts_pos[i] for i,c in enumerate(coords_pos)}
    for i,c in enumerate(coords_neg):
        key = tuple(c)
        pos_dict[key] = pos_dict.get(key, 0) - counts_neg[i]

    pos_collection = np.array([c for c,v in pos_dict.items() if v>0])
    if len(pos_collection) == 0:
        return new_events  # 没有正事件主导像素，直接返回

    # Step 2: 计算矩形和中心
    x1, y1 = pos_collection.min(axis=0)
    x2, y2 = pos_collection.max(axis=0)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    width, height = (x2-x1+1), (y2-y1+1)
    print(f"矩形中心=({cx},{cy}), 宽={width}, 高={height}")

    # Step 3: 高斯函数
    sigma_x, sigma_y = width*sxf, height*syf
    # sigma_x, sigma_y = width * 0.0549, height * 0.0709
    def gaussian(x, y):
        return np.exp(-(((x-cx)**2)/(2*sigma_x**2) + ((y-cy)**2)/(2*sigma_y**2)))

    # Step 4: 找到所有事件涉及的像素（去重）
    unique_pixels, inverse_idx = np.unique(new_events[:,1:3].astype(int), axis=0, return_inverse=True)

    # Step 5: 针对 pos_collection 的像素决定是否保留
    keep_pixels = np.ones(len(unique_pixels), dtype=bool)  # 默认保留
    # 严格逐对比对 (x,y)
    unique_keys = unique_pixels.view([('', unique_pixels.dtype)] * 2).ravel()
    pos_keys = pos_collection.view([('', pos_collection.dtype)] * 2).ravel()
    pos_mask = np.in1d(unique_keys, pos_keys)

    probs = gaussian(unique_pixels[pos_mask, 0], unique_pixels[pos_mask, 1])
    keep_pixels[pos_mask] = (np.random.rand(np.sum(pos_mask)) < probs)

    # Step 6: 映射回事件级别
    mask = keep_pixels[inverse_idx]
    return new_events[mask]
