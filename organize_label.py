import os
import shutil
import re

def organize_labels(root_dir, out_dir, val_seq_ids, test_seq_ids):
    """
    根据序列号分组，将预处理过的标签文件（.npy）从源目录复制到目标目录。

    Args:
        root_dir (str): 包含所有序列文件夹的根目录路径。
        out_dir (str): 输出目录路径，'train', 'val', 'test' 文件夹将被创建于此。
        val_seq_ids (set): 用于验证集的序列号集合。
        test_seq_ids (set): 用于测试集的序列号集合。
    """
    # --- 1. 创建目标文件夹 ---
    # 使用 os.path.join 确保路径在不同操作系统下都能正常工作
    train_dest = os.path.join(out_dir, 'train')
    val_dest = os.path.join(out_dir, 'val')
    test_dest = os.path.join(out_dir, 'test')

    # os.makedirs(..., exist_ok=True) 会创建所有必需的中间目录，且如果目录已存在也不会报错
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)
    
    print(f"目标文件夹已确认/创建于: {out_dir}")

    # --- 2. 检查根目录是否存在 ---
    if not os.path.isdir(root_dir):
        print(f"错误：根目录 '{root_dir}' 不存在，请检查路径。")
        return

    # --- 3. 遍历根目录下的所有序列文件夹 ---
    total_files_copied = 0
    print("\n开始处理序列文件夹...")

    for folder_name in sorted(os.listdir(root_dir)): # sorted() 使处理顺序更可预测
        seq_folder_path = os.path.join(root_dir, folder_name)
        
        # 确保处理的是文件夹，而不是根目录下的零散文件
        if not os.path.isdir(seq_folder_path):
            continue

        # --- 4. 从文件夹名称中提取序列号 ---
        # 使用正则表达式匹配以 'seq' 开头后跟数字的模式
        match = re.match(r'seq(\d+)', folder_name)
        if not match:
            print(f"  - 警告：跳过 '{folder_name}'，无法从中解析出序列号。")
            continue
        
        seq_id = int(match.group(1))

        # --- 5. 根据序列号确定所属的数据集 (train/val/test) ---
        if seq_id in val_seq_ids:
            target_folder = val_dest
            split_name = 'val'
        elif seq_id in test_seq_ids:
            target_folder = test_dest
            split_name = 'test'
        else:
            target_folder = train_dest
            split_name = 'train'
            
        # --- 6. 定位标签文件夹并复制 .npy 文件 ---
        labels_source_dir = os.path.join(seq_folder_path, 'labels_preprocessed')

        if not os.path.isdir(labels_source_dir):
            print(f"  - 警告：在 '{seq_folder_path}' 中未找到 'labels_preprocessed' 文件夹，跳过。")
            continue
            
        print(f"处理序列 {seq_id} ({folder_name}) -> 分配至 '{split_name}' 组")
        
        files_in_dir = os.listdir(labels_source_dir)
        npy_files_found = False
        for filename in files_in_dir:
            if filename.endswith('.npy'):
                npy_files_found = True
                source_file_path = os.path.join(labels_source_dir, filename)
                destination_file_path = os.path.join(target_folder, filename)
                
                # 使用 shutil.copy2 进行“完美copy”，它会复制文件内容和元数据
                shutil.copy2(source_file_path, destination_file_path)
                total_files_copied += 1
        
        if not npy_files_found:
             print(f"  - 警告：在 '{labels_source_dir}' 中未找到任何 .npy 文件。")

    print(f"\n--- 处理完成 ---")
    print(f"总共复制了 {total_files_copied} 个 .npy 文件。")


# --- 程序主入口 ---
if __name__ == '__main__':
    # --- 请在这里配置您的路径和序列号 ---
    
    # 1. 源数据根目录
    # Windows 示例: 'C:/Users/YourUser/Desktop/my_data'
    # Linux/Mac 示例: '/home/user/data/my_data'
    ROOT_DIRECTORY = 'G:/WeChat/Chat_History/xwechat_files/wxid_88ld0q0zlst722_b12b/msg/file/2025-10/vanilla-v2e/seq0001-0090' #里面有90个大场景的文件夹
    # ROOT_DIRECTORY = "E:/xwechat_files/wxid_88ld0q0zlst722_b12b/msg/file/2025-10/blender_labels_gs_test"
    
    # 2. 输出目录
    OUTPUT_DIRECTORY = 'G:/Prophesee/blender_training_vanilla/dataset' #最终的dataset文件夹，里面有train/val/test三个子文件夹
    # OUTPUT_DIRECTORY = 'E:/xwechat_files/wxid_88ld0q0zlst722_b12b/msg/file/2025-10/blender_labels_gs_test-out'
    # 3. 序列号分组
    # 使用集合(set)进行查找会比列表(list)更快，尤其当序列数量很大时
    # VAL_SEQUENCES = {63, 67, 74, 80, 86}
    # TEST_SEQUENCES = {64, 66, 71, 77, 84}
    VAL_SEQUENCES = {2,6,17,24,26,39,51,56,43,32,63,67,74,80,86}
    TEST_SEQUENCES = {4,9,14,20,27,46,53,50,47,36,64,66,71,77,84}

    # --- 执行函数 ---
    organize_labels(ROOT_DIRECTORY, OUTPUT_DIRECTORY, VAL_SEQUENCES, TEST_SEQUENCES)