import os
import math
import subprocess


def get_video_fps(video_path):
    """获取视频帧率"""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of',
        'default=noprint_wrappers=1:nokey=1', video_path
    ]
    fps_str = subprocess.check_output(cmd).decode().strip()
    numerator, denominator = map(float, fps_str.split('/'))
    return numerator / denominator


def get_video_duration(video_path):
    """获取视频总时长（单位：秒）"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',  # 修正参数名
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # 新增捕获stderr
        text=True,
        encoding='utf-8'  # 明确指定编码
    )

    # 优先尝试从stderr获取输出（ffprobe的常规输出通道）
    output = result.stderr.strip() or result.stdout.strip()

    if not output:
        raise ValueError("无法获取视频时长，请检查文件路径或格式")

    return float(output)


def extract_labeled_frames(video_path, time_dict, output_dir="labeled_frames"):
    """提取带标签的帧及时间段外的帧"""
    fps = get_video_fps(video_path)
    # assert fps == 30.0
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频总帧数
    duration = get_video_duration(video_path)
    total_frames = int(duration * fps) - 1  # 帧号从0开始

    # 生成时间段内的所有区间（合并重叠）
    intervals = []
    labels = []
    for clip, label in zip(time_dict["z"], time_dict["av"]):
        labels.append(label.get('1'))
        if len(clip) == 1:
            start, end = clip[0], clip[0]+0.01
            start_frame = math.floor(start * fps)
            end_frame = math.ceil(end * fps)
            intervals.append((start_frame, end_frame))
        else:
            start, end = clip[0], clip[1]
            start_frame = math.floor(start * fps)
            # end_frame = math.ceil(end * fps)
            end_frame = math.floor(end * fps)
            intervals.append((start_frame, end_frame))

    # 合并重叠区间
    intervals.sort()
    merged = []
    merged_labels = []
    for interval, label in zip(intervals, labels):
        if not merged:
            merged.append(interval)
            merged_labels.append(label)
        else:
            last = merged[-1]
            last_label = merged_labels[-1]
            if interval[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], interval[1]))
                assert last_label == label
            else:
                merged.append(interval)
                merged_labels.append(label)

    # 生成时间段外的区间
    outside_intervals = []
    prev_end = -1
    for start, end in merged:
        if start > prev_end + 1:
            outside_intervals.append((prev_end + 1, start - 1))
        prev_end = max(prev_end, end)
    if prev_end < total_frames:
        outside_intervals.append((prev_end + 1, total_frames))

    # 处理时间段外的帧
    for s, e in outside_intervals:
        if s > e: continue  # 跳过无效区间
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=between(n\,{s}\,{e})',
            '-vsync', '0',
            '-start_number', str(s),
            f'{output_dir}/frame_%05d_None.jpg'  # 普通文件名
        ]
        subprocess.run(cmd)

    # 处理时间段内的帧（带标签）
    for idx, (start_frame, end_frame) in enumerate(merged):
        # start_frame = math.floor(start * fps)
        # end_frame = math.ceil(end * fps)
        action = merged_labels[idx]

        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=between(n\,{start_frame}\,{end_frame})',
            '-vsync', '0',
            '-start_number', str(start_frame),
            f'{output_dir}/frame_%05d_{action}.jpg'  # 带动作标签
        ]
        print(f"处理带标签片段：{idx}")
        subprocess.run(cmd)


# 示例用法
time_dict = \
{'z': [[2.402, 3.402], [5.315, 6.315], [7.767, 8.767], [8.815, 9.815], [10.866, 11.866], [13.406, 14.406], [15.005, 16.005], [18.905, 19.905], [35.642, 36.642], [39.179, 40.179], [45.548, 47.17942], [51.624, 52.624], [149.92, 150.92], [151.308, 152.308], [158.674, 159.674], [166.941, 167.941], [170.408, 171.408], [174.707, 175.707], [187.108, 188.108], [190.474, 191.474], [192.141, 193.141], [196.208, 197.208], [201.408, 202.408], [208.574, 209.574], [231.749, 232.749], [239.024, 240.024], [240.79, 241.79], [246.524, 247.524], [255.372, 256.372], [258.826, 259.826], [266.262, 267.262], [274.633, 275.633], [279.633, 280.633], [297.434, 298.434], [322.585, 323.585], [325.769, 326.769], [329.534, 330.534], [331.874, 332.874]], 'av': [{'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'head-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}, {'1': 'face-tic'}]}




output_dir = "D:/dataset/TIC-ori/test/6"
extract_labeled_frames("D:/DesktopFile/zhang/参考文献/抽动症/视频收集2月/视频收集2月/刘宇杰41058038.mp4", time_dict, output_dir)