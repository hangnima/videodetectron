import os
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2Processor


class WavToTensor:
    """
    将WAV文件转换为tensor，并保留时间信息
    
    这个类提供了一系列方法，用于：
    1. 加载WAV文件并转换为PyTorch张量
    2. 生成详细的时间信息（采样率、持续时间、每个采样点的时间戳）
    3. 使用Wav2Vec2模型提取音频特征 
    4. 提取帧级特征用于语音处理
    5. 批量处理WAV文件目录
    """
    def __init__(self, model_path="./ckp/wav2vec2-base-960h", sample_rate=16000):
        """
        初始化WavToTensor类
        
        Args:
            model_path (str): Wav2Vec2处理器路径，用于特征提取
                              如果不需要提取特征，可以设置为None
            sample_rate (int): 目标采样率，标准语音处理通常使用16000Hz
                               所有加载的音频都会被重采样到这个采样率
        """
        self.sample_rate = sample_rate
        
        # 如果指定了模型路径，则加载处理器
        # 该处理器用于将音频转换为Wav2Vec2模型可以处理的特征
        if model_path:
            try:
                print(f"加载Wav2Vec2处理器: {model_path}")
                self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            except Exception as e:
                print(f"加载处理器时出错: {e}")
                self.processor = None
        else:
            self.processor = None

    def load_wav(self, wav_path, return_tensor=True):
        """
        加载WAV文件并转换为numpy数组或tensor
        
        该方法使用librosa库加载音频，支持各种采样率的WAV文件
        如果原始采样率与目标采样率不同，会自动进行重采样
        
        Args:
            wav_path (str): WAV文件路径
            return_tensor (bool): 是否返回PyTorch张量
                                 如果为False，则返回numpy数组
            
        Returns:
            tuple: (audio_data, sr) 
                - audio_data: 音频数据（numpy数组或tensor）形状为[L]，其中L是采样点数
                - sr: 原始采样率（整数）
                
        注意:
            即使return_tensor=False，采样率仍会调整为self.sample_rate
            返回的sr是原始采样率，用于记录
        """
        try:
            # 使用原始采样率加载音频，不改变通道数（默认转为单声道）
            y, sr = librosa.load(wav_path, sr=None)  # 使用原始采样率加载
            original_sr = sr  # 保存原始采样率
            
            # 如果原始采样率不等于目标采样率，进行重采样
            if sr != self.sample_rate:
                print(f"将采样率从 {sr} Hz 重采样到 {self.sample_rate} Hz")
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            # 如果需要，将numpy数组转换为PyTorch张量
            if return_tensor:
                y = torch.tensor(y)
                
            return y, original_sr
        except Exception as e:
            print(f"加载WAV文件时出错: {e}")
            return None, None

    def create_time_info(self, audio_data, sr):
        """
        为音频数据创建详细的时间信息
        
        该方法计算音频总时长，并为每个采样点生成时间戳
        
        Args:
            audio_data: 音频数据（numpy数组或tensor）
            sr (int): 采样率，用于计算时间
            
        Returns:
            dict: 包含时间信息的字典，结构如下：
                {
                    "sample_rate": 采样率（整数，如16000Hz）,
                    "duration": 总时长（秒，浮点数）,
                    "num_samples": 采样点总数（整数）,
                    "timestamps": 每个采样点的时间戳（与audio_data同类型的数组）
                }
        """
        # 获取音频长度（采样点数）
        if isinstance(audio_data, torch.Tensor):
            audio_length = audio_data.shape[0]
        else:
            audio_length = len(audio_data)
        
        # 计算总时长（秒） = 采样点数 / 采样率
        duration = audio_length / sr
        
        # 为每个采样点创建时间戳：从0秒到总时长，均匀分布
        # 保持与输入相同的数据类型（tensor或numpy）
        if isinstance(audio_data, torch.Tensor):
            timestamps = torch.linspace(0, duration, audio_length)
        else:
            timestamps = np.linspace(0, duration, audio_length)
        
        # 收集所有时间信息到一个字典中
        time_info = {
            "sample_rate": sr,            # 采样率
            "duration": duration,         # 总时长（秒）
            "num_samples": audio_length,  # 采样点总数
            "timestamps": timestamps      # 每个采样点的时间戳
        }
        
        return time_info

    def wav_to_features(self, wav_path, chunk_size=None):
        """
        将WAV文件转换为Wav2Vec2特征，并保留时间信息
        
        该方法使用Wav2Vec2处理器将音频转换为模型特征
        对于较长的音频，支持分块处理以避免内存不足
        
        Args:
            wav_path (str): WAV文件路径
            chunk_size (int): 分块处理的大小（采样点数）
                             如果为None则一次性处理整个文件
                             对于长音频，建议设置为16000*30（30秒音频）
            
        Returns:
            tuple: (features, time_info)
                - features: 特征张量，形状为[1, L, hidden_size]
                - time_info: 时间信息字典，与create_time_info返回格式相同
                            如果分块处理，额外包含chunk_timestamps字段
        
        注意:
            此方法需要已初始化的Wav2Vec2处理器，否则返回None
        """
        # 检查处理器是否已初始化
        if not self.processor:
            print("未加载处理器，无法提取特征")
            return None, None
        
        # 加载音频 - 这里使用return_tensor=False获取numpy数组
        # 因为Wav2Vec2处理器需要numpy输入
        audio_data, original_sr = self.load_wav(wav_path, return_tensor=False)
        if audio_data is None:
            return None, None
        
        # 创建时间信息
        time_info = self.create_time_info(audio_data, self.sample_rate)
        
        # 如果指定了分块大小，且音频长度超过分块大小，则分块处理
        # 这对于处理长音频很有用，可以避免内存不足
        if chunk_size and len(audio_data) > chunk_size:
            features_list = []       # 存储每个块的特征
            timestamps_list = []     # 存储每个块对应的时间戳
            
            # 按块遍历音频
            for i in range(0, len(audio_data), chunk_size):
                # 确保不超出音频范围
                end_idx = min(i + chunk_size, len(audio_data))
                print(f"处理音频段: {i/self.sample_rate:.1f}s - {end_idx/self.sample_rate:.1f}s")
                
                # 提取当前块
                chunk = audio_data[i:end_idx]
                
                # 使用处理器处理当前块
                inputs = self.processor(chunk, sampling_rate=self.sample_rate, return_tensors="pt")
                features_list.append(inputs.input_values)
                
                # 保存对应的时间戳，用于后续时间对齐
                chunk_timestamps = time_info["timestamps"][i:end_idx]
                timestamps_list.append(chunk_timestamps)
            
            # 合并所有块的特征和时间戳
            features = torch.cat(features_list, dim=1)  # 在序列维度（dim=1）上拼接
            time_info["chunk_timestamps"] = timestamps_list  # 添加块时间戳到时间信息
        else:
            # 一次性处理整个文件
            inputs = self.processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt")
            features = inputs.input_values
        
        return features, time_info

    def process_directory(self, wav_dir, output_dir=None, return_results=False, extract_frames=False):
        """
        处理目录中的所有WAV文件，提取张量和时间信息
        
        该方法可以批量处理整个目录中的WAV文件，可选择保存结果到磁盘
        
        Args:
            wav_dir (str): WAV文件目录
            output_dir (str): 输出目录，如果为None则不保存结果
            return_results (bool): 是否返回处理结果的字典
                                  对于大量文件，设为False可以节省内存
            extract_frames (bool): 是否提取帧级特征，默认为False
                                  设为True时会为每个文件提取与视频对齐的帧级特征（30帧/秒）
            
        Returns:
            dict: WAV文件名到处理结果的映射（如果return_results为True）
                 处理结果包含音频张量、时间信息、可能的特征和帧级张量
        """
        # 确保输出目录存在
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取所有WAV文件
        wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
        if not wav_files:
            print(f"在 {wav_dir} 中没有找到WAV文件")
            return {}
        
        # 用于存储处理结果的字典
        results = {}
        
        # 使用tqdm显示进度条，处理每个WAV文件
        for wav_file in tqdm(wav_files, desc="处理WAV文件"):
            wav_path = os.path.join(wav_dir, wav_file)
            
            try:
                print(f"\n处理WAV文件: {wav_file}")
                
                # 1. 加载音频数据并转换为tensor
                audio_tensor, original_sr = self.load_wav(wav_path)
                
                # 2. 创建时间信息
                time_info = self.create_time_info(audio_tensor, self.sample_rate)
                
                # 3. 如果有处理器，提取特征
                if self.processor:
                    features, _ = self.wav_to_features(wav_path)
                    time_info["has_features"] = features is not None
                else:
                    features = None
                    time_info["has_features"] = False
                
                # 4. 提取帧级特征（如果需要）
                frames_tensor = None
                frame_timestamps = None
                if extract_frames:
                    print(f"提取帧级特征（30帧/秒）...")
                    frames_tensor, frame_timestamps = self.get_frame_level_tensors(
                        wav_path, frame_duration=1.0/10, frame_shift=1.0/10
                    )
                    if frames_tensor is not None:
                        print(f"提取了 {frames_tensor.shape[0]} 帧，对应约 {10:.2f} 帧/秒")
                
                # 5. 构建结果字典
                result = {
                    "audio_tensor": audio_tensor,    # 音频张量
                    "time_info": time_info,          # 时间信息 
                    "original_sr": original_sr       # 原始采样率
                }
                
                # 如果成功提取了特征，添加到结果中
                if features is not None:
                    result["features"] = features
                
                # 如果成功提取了帧级特征，添加到结果中
                if frames_tensor is not None:
                    result["frames_tensor"] = frames_tensor
                    result["frame_timestamps"] = frame_timestamps
                
                # 6. 保存结果到文件（如果指定了输出目录）
                if output_dir:
                    output_file = os.path.join(output_dir, f"{os.path.splitext(wav_file)[0]}_tensor.pt")
                    torch.save(result, output_file)
                    print(f"已保存到: {output_file}")
                
                # 7. 如果要求返回结果，添加到结果字典
                if return_results:
                    results[wav_file] = result
                
                # 8. 打印处理信息
                print(f"音频长度: {len(audio_tensor)} 采样点 ({time_info['duration']:.2f}秒)")
                print(f"采样率: {time_info['sample_rate']} Hz (原始: {original_sr} Hz)")
                
            except Exception as e:
                print(f"处理 {wav_file} 时出错: {e}")
        
        # 根据return_results参数决定是否返回处理结果
        return results if return_results else None

    def get_frame_level_tensors(self, wav_path, frame_duration=1/10, frame_shift=1/10):
        """
        获取帧级别的张量表示，用于语音处理中的帧级分析
        
        该方法将音频分割成一系列帧，可以配置为与视频帧对齐
        
        Args:
            wav_path (str): WAV文件路径
            frame_duration (float): 每帧的持续时间（秒）
                                   默认值为0.1秒（100ms），与10帧/秒视频的帧持续时间匹配
            frame_shift (float): 帧移（秒），即相邻帧之间的时间间隔
                                默认值为0.1秒（100ms），对应10帧/秒的视频
            
        Returns:
            tuple: (frames_tensor, frame_timestamps)
                - frames_tensor: 帧级张量，形状为 [num_frames, frame_size]
                  其中num_frames是帧数，frame_size是每帧的采样点数
                - frame_timestamps: 每帧的中心时间戳，形状为 [num_frames]
        """
        # 1. 加载音频
        audio_data, sr = self.load_wav(wav_path, return_tensor=False)
        if audio_data is None:
            return None, None
        
        # 2. 计算帧大小和帧移（以采样点为单位）
        # 帧大小 = 帧持续时间 × 采样率
        frame_size = int(frame_duration * self.sample_rate)
        # 帧移（采样点数） = 帧移（秒） × 采样率
        frame_shift_samples = int(frame_shift * self.sample_rate)
        
        # 3. 计算总帧数
        # 公式解释：(总长度 - 帧大小) / 帧移 + 1，向下取整
        num_frames = max(0, 1 + (len(audio_data) - frame_size) // frame_shift_samples)
        
        # 如果音频太短，无法提取完整帧
        if num_frames <= 0:
            print(f"音频太短，无法提取帧: {wav_path}")
            return None, None
        
        # 4. 初始化存储帧和时间戳的数组
        frames = np.zeros((num_frames, frame_size))
        frame_timestamps = np.zeros(num_frames)
        
        # 5. 提取每一帧数据和对应的时间戳
        for i in range(num_frames):
            # 计算当前帧的起始和结束位置
            start_idx = i * frame_shift_samples
            end_idx = start_idx + frame_size
            
            # 确保不超出音频范围
            if end_idx <= len(audio_data):
                # 复制当前帧的数据
                frames[i] = audio_data[start_idx:end_idx]
                
                # 计算帧中心的时间戳：起始点 + 帧大小/2，再除以采样率得到秒数
                frame_timestamps[i] = (start_idx + frame_size/2) / self.sample_rate
        
        # 6. 转换为PyTorch张量
        frames_tensor = torch.tensor(frames, dtype=torch.float32)
        frame_timestamps_tensor = torch.tensor(frame_timestamps, dtype=torch.float32)
        
        # 7. 打印处理信息
        print(f"提取了 {num_frames} 帧，每帧 {frame_size} 采样点")
        print(f"帧级张量形状: {frames_tensor.shape}")
        
        return frames_tensor, frame_timestamps_tensor


def demo():
    """
    演示如何使用WavToTensor类的各种功能
    
    该函数展示了四个主要示例：
    1. 处理单个WAV文件并获取基本张量
    2. 提取帧级特征
    3. 提取Wav2Vec特征
    4. 批量处理整个目录
    
    这些示例可以作为使用本模块的参考代码
    """
    # 初始化处理器
    wav_processor = WavToTensor(model_path="./ckp/wav2vec2-base-960h")
    
    # 设置音频目录和输出目录
    audio_dir = "D:/DesktopFile/zhang/参考文献/抽动症/视频收集2月/音频"
    output_dir = "D:/DesktopFile/zhang/参考文献/抽动症/视频收集2月/tensor"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取音频文件列表
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    if wav_files:
        # 选择第一个WAV文件作为示例
        sample_wav = os.path.join(audio_dir, wav_files[0])
        
        #========== 示例1: 处理单个WAV文件 ===========
        print(f"\n示例1: 处理单个WAV文件 {sample_wav}")
        
        # 加载WAV文件并获取tensor
        audio_tensor, sr = wav_processor.load_wav(sample_wav)
        time_info = wav_processor.create_time_info(audio_tensor, wav_processor.sample_rate)
        
        # 打印基本信息
        print(f"音频tensor形状: {audio_tensor.shape}")
        print(f"音频时长: {time_info['duration']:.2f}秒")
        print(f"采样率: {time_info['sample_rate']} Hz")
        
        # 保存基本张量
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(sample_wav))[0]}_basic.pt")
        torch.save({"audio_tensor": audio_tensor, "time_info": time_info}, output_file)
        print(f"已保存基本tensor到: {output_file}")
        
        #========== 示例2: 提取帧级特征 ===========
        print(f"\n示例2: 提取帧级特征")
        # 使用与视频对齐的参数：33ms帧长，33.3ms帧移（对应30帧/秒的视频）
        frames_tensor, frame_timestamps = wav_processor.get_frame_level_tensors(
            sample_wav, frame_duration=0.1, frame_shift=0.1
        )
        
        if frames_tensor is not None:
            # 保存帧级特征
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(sample_wav))[0]}_frames.pt")
            torch.save({"frames_tensor": frames_tensor, "frame_timestamps": frame_timestamps}, output_file)
            print(f"已保存帧级特征到: {output_file}")
            print(f"帧率: 约 {1.0/0.1:.2f} 帧/秒，适合与10帧/秒视频对齐")
        
        #========== 示例3: 提取Wav2Vec特征 ===========
        if wav_processor.processor:
            print(f"\n示例3: 提取wav2vec特征")
            features, feature_time_info = wav_processor.wav_to_features(sample_wav)
            
            if features is not None:
                # 保存Wav2Vec特征
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(sample_wav))[0]}_features.pt")
                torch.save({"features": features, "time_info": feature_time_info}, output_file)
                print(f"已保存wav2vec特征到: {output_file}")
                print(f"特征形状: {features.shape}")
    
    #========== 示例4: 批量处理目录 ===========
    print(f"\n示例4: 批量处理目录")
    results = wav_processor.process_directory(audio_dir, output_dir, return_results=True, extract_frames=True)
    print(f"已处理 {len(results)} 个WAV文件")
    print(f"提取了帧级特征，与10帧/秒的视频对齐")


# 当直接运行此脚本时，执行演示函数
if __name__ == "__main__":
    demo() 