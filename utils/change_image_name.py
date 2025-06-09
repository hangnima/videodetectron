import os

def replace_none_with_jump(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否符合模式 frame_xxxx_None.jpg
        if filename.startswith("frame_") and filename.endswith("_head-tic.jpg"):
            # 替换字符串中的 "None" 为 "jump"
            new_filename = filename.replace("_head-tic", "_face-tic")
            # 构建旧文件和新文件的完整路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            # 执行重命名操作
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 调用示例
replace_none_with_jump("D:/code/videodetectron/data/TIC/test/1")
