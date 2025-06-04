import os


def save_folder_paths(root_dir, output_file):
    """
    保存所有1-9文件夹下的batch子文件夹路径到文本文件

    参数:
    root_dir: 包含1-9文件夹的根目录
    output_file: 输出文本文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历1-9文件夹
        for folder_num in range(1, 10):
            num_dir = os.path.join(root_dir, str(folder_num))

            # 检查数字文件夹是否存在
            if not os.path.exists(num_dir):
                print(f"警告: 文件夹 {num_dir} 不存在")
                continue

            # 遍历数字文件夹下的所有子文件夹
            for batch_folder in os.listdir(num_dir):
                batch_path = os.path.join(num_dir, batch_folder)

                # 只处理文件夹，忽略文件
                if os.path.isdir(batch_path):
                    # 创建相对路径格式: 1/batch_1
                    relative_path = f"{folder_num}/{batch_folder}"
                    f.write(relative_path + '\n')
                    print(f"添加路径: {relative_path}")


if __name__ == "__main__":
    # 配置路径
    root_directory = "D:/code/videodetectron/data/TIC_new/train/"  # 替换为包含1-9文件夹的根目录
    output_filename = "train_tic.txt"  # 输出文件名

    # 执行保存操作
    save_folder_paths(root_directory, output_filename)
    print(f"\n所有路径已保存到: {output_filename}")