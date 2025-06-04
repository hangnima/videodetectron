import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm  # 用于显示进度条

def convert_mp4_to_wav(video_dir, audio_dir):
    """
    将 video 目录下的所有 mp4 文件转换为 audio 目录下的 wav 文件。

    Args:
        video_dir (str): 包含 mp4 文件的目录。
        audio_dir (str): 输出 wav 文件的目录。
    """

    # 确保 audio 目录存在
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    if not mp4_files:
        print(f"在 {video_dir} 中没有找到 mp4 文件。")
        return

    for mp4_file in tqdm(mp4_files, desc="转换文件"):
        mp4_path = os.path.join(video_dir, mp4_file)
        wav_file = mp4_file.replace(".mp4", ".wav")
        wav_path = os.path.join(audio_dir, wav_file)

        try:
            # 使用 librosa 加载音频，自动处理 mp4
            y, sr = librosa.load(mp4_path, sr=None, mono=False) # 加载音频，保持原始采样率和声道数

            # 转换为单声道 (如果不是单声道)
            if len(y.shape) > 1:
                y = librosa.to_mono(y)

            # 重采样到 16kHz
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)

            # 使用 soundfile 保存为 16-bit PCM WAV
            sf.write(wav_path, y_resampled, 16000, subtype='PCM_16')


            # 可选: 加载 WAV 文件到 NumPy 数组
            audio_numpy = librosa.load(wav_path, sr=16000, mono=True)[0]  # librosa.load会自动转换为单声道

            # 可选: 加载 WAV 文件到 PyTorch 张量
            #audio_tensor = torch.tensor(audio_numpy)


        except Exception as e:
            print(f"转换 {mp4_file} 时发生错误: {e}")

        # 示例：打印一些信息 (可以删除)
        # print(f"已转换: {mp4_file} -> {wav_file}")
        # print(f"  采样率: {sr} Hz -> 16000 Hz")
        # print(f"  声道数: {len(y.shape) > 1 if y.ndim > 1 else 1} -> 1 (单声道)")
        # print(f"  NumPy 数组形状: {audio_numpy.shape}")
        # print(f"  PyTorch 张量形状: {audio_tensor.shape}")


# 示例用法：
video_dir = "D:/DesktopFile/zhang/参考文献/抽动症/视频收集2月/视频收集2月"  # 你的 video 目录
audio_dir = "D:/DesktopFile/zhang/参考文献/抽动症/视频收集2月/音频"  # 你想要保存 wav 文件的目录
convert_mp4_to_wav(video_dir, audio_dir)

print("转换完成！")
