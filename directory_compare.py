import os
import sys
from pathlib import Path


def get_files_without_extension(directory):
    """
    获取目录下所有文件（包括子目录），返回不带扩展名的文件名集合
    """
    files_set = set()
    
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return files_set
    
    try:
        for root, dirs, files in os.walk(directory):
            # 获取相对于基础目录的路径
            rel_root = os.path.relpath(root, directory)
            if rel_root == '.':
                rel_root = ''
            
            # 添加所有子目录
            for dir_name in dirs:
                if rel_root:
                    dir_path = os.path.join(rel_root, dir_name)
                else:
                    dir_path = dir_name
                files_set.add(('dir', dir_path))
            
            # 添加所有文件（不含扩展名）
            for file_name in files:
                # 去掉扩展名
                name_without_ext = os.path.splitext(file_name)[0]
                if rel_root:
                    file_path = os.path.join(rel_root, name_without_ext)
                else:
                    file_path = name_without_ext
                files_set.add(('file', file_path))
                
    except PermissionError as e:
        print(f"权限错误: 无法访问某些文件或目录 - {e}")
    except Exception as e:
        print(f"遍历目录时发生错误: {e}")
    
    return files_set


def compare_directories(dir_a, dir_b):
    """
    比较两个目录，找出B目录中缺失的文件和目录
    """
    print(f"正在比较目录:")
    print(f"A目录: {os.path.abspath(dir_a)}")
    print(f"B目录: {os.path.abspath(dir_b)}")
    print("-" * 50)
    
    # 获取两个目录的文件集合
    files_a = get_files_without_extension(dir_a)
    files_b = get_files_without_extension(dir_b)
    
    if not files_a and not files_b:
        print("两个目录都为空或无法访问")
        return
    
    # 找出A目录有但B目录没有的文件
    missing_in_b = files_a - files_b
    
    if not missing_in_b:
        print("✅ B目录包含了A目录的所有文件和目录!")
        return
    
    # 分别统计缺失的文件和目录
    missing_files = []
    missing_dirs = []
    
    for item_type, item_path in missing_in_b:
        if item_type == 'file':
            missing_files.append(item_path)
        else:
            missing_dirs.append(item_path)
    
    # 输出结果
    print(f"❌ B目录缺失以下内容 (共 {len(missing_in_b)} 项):")
    print()
    
    if missing_dirs:
        print(f"缺失的目录 ({len(missing_dirs)} 个):")
        for dir_path in sorted(missing_dirs):
            print(f"  📁 {dir_path}")
        print()
    
    if missing_files:
        print(f"缺失的文件 ({len(missing_files)} 个):")
        for file_path in sorted(missing_files):
            print(f"  📄 {file_path}")


def main():
    """
    主函数
    """
    print("目录文件比较工具")
    print("=" * 50)
    
    # 获取目录路径
    if len(sys.argv) >= 3:
        dir_a = sys.argv[1]
        dir_b = sys.argv[2]
    else:
        # 交互式输入
        dir_a = input("请输入A目录路径: ").strip()
        dir_b = input("请输入B目录路径: ").strip()
    
    # 检查目录是否存在
    if not os.path.exists(dir_a):
        print(f"错误: A目录 '{dir_a}' 不存在")
        return
    
    if not os.path.exists(dir_b):
        print(f"错误: B目录 '{dir_b}' 不存在")
        return
    
    if not os.path.isdir(dir_a):
        print(f"错误: '{dir_a}' 不是一个目录")
        return
    
    if not os.path.isdir(dir_b):
        print(f"错误: '{dir_b}' 不是一个目录")
        return
    
    # 执行比较
    try:
        compare_directories(dir_a, dir_b)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")


if __name__ == "__main__":
    main()

