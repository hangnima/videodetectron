import os
import shutil


def sync_and_clean_folders(folder_a, folder_b, exclude_extensions=None, dry_run=False):
    """
    同步清理两个文件夹中的差异图片文件
    :param folder_a: 第一个文件夹路径
    :param folder_b: 第二个文件夹路径
    :param exclude_extensions: 要排除的文件后缀列表（如 ['.tmp']）
    :param dry_run: 试运行模式（仅显示将要删除的文件）
    """
    # 获取图片文件名集合（忽略排除后缀）
    img_ext = {'.png', '.jpg', '.jpeg'}
    exclude = set(exclude_extensions or [])

    files_a = {f for f in os.listdir(folder_a)
               if os.path.splitext(f)[1].lower() in img_ext - exclude}
    files_b = {f for f in os.listdir(folder_b)
               if os.path.splitext(f)[1].lower() in img_ext - exclude}

    # 计算双向差异
    only_in_a = files_a - files_b
    only_in_b = files_b - files_a

    # 构建待删除文件路径列表
    to_delete = [
        *[os.path.join(folder_a, f) for f in only_in_a],
        *[os.path.join(folder_b, f) for f in only_in_b]
    ]

    # 显示即将删除的文件
    if to_delete:
        print("以下文件将被删除：")
        print("\n".join(to_delete))

        # 安全确认机制
        if not dry_run:
            confirm = input(f"确认删除以上 {len(to_delete)} 个文件？(y/n): ")
            if confirm.lower() != 'y':
                print("操作已取消")
                return

    # 执行删除操作
    deleted_count = 0
    for file_path in to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"删除失败：{file_path} - {str(e)}")

    print(f"操作完成，成功删除 {deleted_count}/{len(to_delete)} 个文件")


# 使用示例（试运行模式）
sync_and_clean_folders(
    folder_a="D:/code/videodetectron/data/TIC/train/5_process",
    folder_b="D:/code/videodetectron/data/TIC/train/5_skeleton",
    exclude_extensions=['.bak'],  # 排除.bak备份文件
    dry_run=True  # 设为False执行真实删除
)
