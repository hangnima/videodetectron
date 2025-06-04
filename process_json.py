# import json
#
# def print_json_structure(data, indent=0):
#     """递归打印 JSON 键值结构"""
#     if isinstance(data, dict):
#         for key, value in data.items():
#             print(' ' * indent + f'Key: {key}')
#             if isinstance(value, (dict, list)):
#                 print_json_structure(value, indent + 4)
#             else:
#                 print(' ' * (indent + 4) + f'Value: {value}')
#     elif isinstance(data, list):
#         for index, item in enumerate(data):
#             print(' ' * indent + f'Index [{index}]')
#             print_json_structure(item, indent + 4)
#
# # 读取 JSON 文件
# file_path = '1.json'  # 替换为你的文件路径
# try:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#     print(f"Successfully loaded JSON from {file_path}\n")
#     print_json_structure(json_data)
# except Exception as e:
#     print(f"Error: {str(e)}")

import json


def extract_z_av_from_metadata(json_path):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取metadata字段
    metadata = data.get('metadata', {})

    # 构建结果字典 {z: [...], av: [...]}
    result = {'z': [], 'av': []}

    # 遍历所有子项
    for item_key, item_value in metadata.items():
        # 提取z和av的值
        z_value = item_value.get('z', [])
        av_value = item_value.get('av', {})

        result['z'].append(z_value)
        result['av'].append(av_value)

    return result


# 使用示例
# json_path = '张梓芃67316304.json'
json_path = 'D:/DesktopFile/zhang/参考文献/抽动症/视频收集2月/label已处理/刘宇杰41058038.json'
output = extract_z_av_from_metadata(json_path)
print(output)
# 筛选符合条件的索引和值
# -------第一种json处理，选一个--------
# filtered_data = [
#     (z_val, av_val)
#     for z_val, av_val in zip(output['z'], output['av'])
#     if '/' not in av_val['1']
# ]

# -------第二种json处理，选一个--------
filtered_data = [
    (z_val, {k: v.split('/', 1)[0] for k, v in av_val.items()})
    for z_val, av_val in zip(output['z'], output['av'])
]

# 构建新字典
new_dict = {
    'z': [item[0] for item in filtered_data],
    'av': [item[1] for item in filtered_data]
}
print(new_dict)