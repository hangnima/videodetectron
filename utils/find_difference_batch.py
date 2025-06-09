import os
import argparse


def find_mismatched_images(root_dir1, root_dir2, extensions=('.jpg', '.jpeg', '.png')):
    """
    比较两个根目录下对应批次文件夹中的图片文件，找出命名不一致的图片

    参数:
    root_dir1: 第一个根目录路径 (如 '1')
    root_dir2: 第二个根目录路径 (如 '1_sk')
    extensions: 图片文件扩展名列表
    """
    # 获取两个目录下的批次文件夹列表
    batches1 = [d for d in os.listdir(root_dir1)
                if os.path.isdir(os.path.join(root_dir1, d))]
    batches2 = [d for d in os.listdir(root_dir2)
                if os.path.isdir(os.path.join(root_dir2, d))]

    # 找出共同的批次文件夹
    common_batches = set(batches1) & set(batches2)
    print(f"找到 {len(common_batches)} 个共同的批次文件夹")

    # 存储所有不匹配的文件
    all_mismatches = []

    # 遍历每个共同批次
    for batch in common_batches:
        batch_path1 = os.path.join(root_dir1, batch)
        batch_path2 = os.path.join(root_dir2, batch)

        # 获取两个批次中的图片文件列表
        images1 = {f for f in os.listdir(batch_path1)
                   if f.lower().endswith(extensions)}
        images2 = {f for f in os.listdir(batch_path2)
                   if f.lower().endswith(extensions)}

        # 找出差异
        only_in_1 = images1 - images2
        only_in_2 = images2 - images1

        if only_in_1 or only_in_2:
            mismatch_info = {
                "batch": batch,
                "only_in_1": sorted(only_in_1),
                "only_in_2": sorted(only_in_2),
                "count1": len(images1),
                "count2": len(images2)
            }
            all_mismatches.append(mismatch_info)

    return all_mismatches


def print_mismatches(mismatches, root_dir1, root_dir2):
    """格式化打印不匹配信息"""
    if not mismatches:
        print("所有批次文件夹中的图片文件完全一致！")
        return

    print(f"\n发现 {len(mismatches)} 个批次存在不一致的图片文件:")
    print("=" * 80)

    for i, mismatch in enumerate(mismatches, 1):
        print(f"{i}. 批次: {mismatch['batch']}")
        print(f"   {root_dir1} 中有 {mismatch['count1']} 个图片文件")
        print(f"   {root_dir2} 中有 {mismatch['count2']} 个图片文件")

        if mismatch['only_in_1']:
            print(f"   仅在 {root_dir1} 中存在的文件:")
            for j, filename in enumerate(mismatch['only_in_1'], 1):
                print(f"     {j}. {filename}")

        if mismatch['only_in_2']:
            print(f"   仅在 {root_dir2} 中存在的文件:")
            for j, filename in enumerate(mismatch['only_in_2'], 1):
                print(f"     {j}. {filename}")

        print("-" * 80)


if __name__ == "__main__":
    for i in range(9,10):
        dir1 = f'D:/code/videodetectron/data/TIC_new/train/{i}'
        dir2 = f'D:/code/videodetectron/data/TIC_new/train/{i}_skeleton'

        # 确保目录存在
        if not os.path.isdir(dir1):
            print(f"错误: 目录 {dir1} 不存在")
            exit(1)

        if not os.path.isdir(dir2):
            print(f"错误: 目录 {dir2} 不存在")
            exit(1)

        # 查找不匹配的文件
        mismatches = find_mismatched_images(dir1, dir2)

        # 打印结果
        print_mismatches(mismatches, dir1, dir2)
