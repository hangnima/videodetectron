import os
import shutil


def extract_frames(src_dir, dst_dir, interval=3):
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)

    # 获取所有符合条件的文件并排序
    files = sorted([f for f in os.listdir(src_dir)
                    if f.startswith("frame_") and f.endswith(".jpg")],
                   key=lambda x: int(x.split("_")[1]))  # 按数字部分排序

    # 初始化计数器
    new_index = 0

    # 遍历文件列表，每隔interval个文件取一个
    for idx, filename in enumerate(files[::interval]):
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.rsplit('_', 1)
        # 构造新文件名
        new_name = f"frame_{str(new_index).zfill(5)}_{parts[1]}.jpg"

        # 源文件和目标路径
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)

        # 复制文件到新目录
        shutil.copy(src_path, dst_path)
        new_index += 1


# 使用示例
src_folder = "D:/dataset/TIC-ori/test/6"
dst_folder = "D:/code/videodetectron/data/TIC/train/6"
extract_frames(src_folder, dst_folder, interval=3)