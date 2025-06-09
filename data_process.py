import os
import shutil
import re
import numpy as np
from collections import defaultdict


def process_images(source_dir, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 1. 收集所有图像文件
    tic_files = defaultdict(list)  # 按帧号分组存储tic图像
    none_files = {}  # 存储None图像，键为帧号

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.wav')):
            continue

        # 提取帧号、类型和扩展名
        match = re.match(r'frame_(\d+)_(.*)\.([^.]+)$', filename)
        if not match:
            continue

        frame_number = int(match.group(1))
        image_type = match.group(2)
        extension = match.group(3).lower()

        if 'tic' in image_type:
            # 带tic的图像
            tic_files[frame_number].append((image_type, filename))
        elif 'None' in image_type:
            # None图像
            none_files[frame_number] = filename

    # 2. 提取所有有tic的帧号并排序
    tic_frame_numbers = sorted(tic_files.keys())

    if not tic_frame_numbers:
        print("没有找到带'tic'的图像，程序退出")
        return

    # 3. 找出连续的tic批次
    batches = []  # 存储每个批次的帧号范围
    current_batch = [tic_frame_numbers[0]]

    for i in range(1, len(tic_frame_numbers)):
        # 检查是否连续
        if tic_frame_numbers[i] == tic_frame_numbers[i - 1] + 1:
            current_batch.append(tic_frame_numbers[i])
        else:
            if len(current_batch) <= 5:
                current_batch = [tic_frame_numbers[i]]
                continue
            # 当前批次结束，保存并开始新批次
            batches.append(current_batch)
            current_batch = [tic_frame_numbers[i]]

    # 添加最后一个批次
    if len(current_batch) > 5:
        batches.append(current_batch)

    print(f"找到 {len(batches)} 个连续的tic批次:")
    for i, batch in enumerate(batches):
        print(f"  批次 {i + 1}: 帧号 {min(batch)} 到 {max(batch)}, 共 {len(batch)} 帧")

    # 4. 处理每个批次
    all_none_frames = sorted(none_files.keys())  # 所有None帧号排序

    for batch_idx, tic_batch in enumerate(batches):
        batch_size = len(tic_batch)
        batch_dir = os.path.join(target_dir, f"batch_{batch_idx + 1}")
        batch_dir_none = os.path.join(target_dir, f"batch_{batch_idx + 1 + len(batches)}")
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(batch_dir_none, exist_ok=True)

        # 5. 复制本批次的tic图像
        for frame in tic_batch:
            for img_type, filename in tic_files[frame]:
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(batch_dir, filename)
                shutil.copy2(src_path, dst_path)

        # 6. 找到连续None序列
        # 计算所有可能的连续区间
        continuous_ranges = []
        start_index = 0

        for i in range(1, len(all_none_frames)):
            # 检查是否连续
            if all_none_frames[i] != all_none_frames[i - 1] + 1:
                # 保存当前连续序列
                continuous_ranges.append(all_none_frames[start_index:i])
                start_index = i

        # 添加最后一个序列
        continuous_ranges.append(all_none_frames[start_index:])

        # 找到长度至少为batch_size的连续序列
        selected_none_frames = []
        selected_range = []

        for seq in continuous_ranges:
            if len(seq) >= batch_size:
                # 取第一个满足条件的连续序列
                selected_none_frames = seq[:batch_size]
                selected_range = seq
                break

        if not selected_none_frames:
            print(f"警告: 批次 {batch_idx + 1} 找不到连续 {batch_size} 个'None'图像")
            continue

        # 7. 复制选中的None图像
        for frame in selected_none_frames:
            if frame in none_files:
                src_path = os.path.join(source_dir, none_files[frame])
                dst_path = os.path.join(batch_dir_none, none_files[frame])
                shutil.copy2(src_path, dst_path)

        # 8. 从全局None列表中移除已使用的帧
        for frame in selected_none_frames:
            if frame in all_none_frames:
                all_none_frames.remove(frame)

        print(f"批次 {batch_idx + 1}: 使用None帧 {min(selected_none_frames)} 到 {max(selected_none_frames)}")

    # 9. 打印最终统计信息
    total_tic_images = sum(len(imgs) for imgs in tic_files.values())
    print(f"\n处理完成!")
    print(f"总共处理了 {len(batches)} 个批次")
    print(f"包含 {total_tic_images} 个tic图像")
    print(f"使用了 {sum(len(batch) for batch in batches)} 个None图像")
    print(f"剩余 {len(all_none_frames)} 个未使用的None图像")


if __name__ == "__main__":
    # 配置路径
    for i in range(9,10):
        source_directory = f"/data/hym/tic3/videodetectron/data/TIC/train/{i}_wav"  # 替换为您的源目录路径
        target_directory = f"/data/hym/tic3/videodetectron/data/TIC/train/{i}_wav"  # 替换为目标目录路径

        process_images(source_directory, target_directory)