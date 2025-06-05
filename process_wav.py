import os
import wave
import numpy as np
from pathlib import Path
import argparse

def get_jpg_files(jpg_dir):
    """获取指定目录下的所有jpg文件"""
    jpg_files = []
    if os.path.exists(jpg_dir):
        for file in os.listdir(jpg_dir):
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(file)
    jpg_files.sort()
    return jpg_files

def slice_wav_file(wav_path, output_dir, jpg_files):
    """根据jpg文件数量切片wav文件"""
    if not jpg_files:
        return False
    
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # 转换为numpy数组
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                return False
            
            audio_data = np.frombuffer(frames, dtype=dtype)
            
            # 立体声处理
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2)
            
            # 计算切片长度
            total_samples = len(audio_data)
            slice_length = total_samples // len(jpg_files)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成切片
            for i, jpg_file in enumerate(jpg_files):
                start_sample = i * slice_length
                if i == len(jpg_files) - 1:
                    end_sample = total_samples
                else:
                    end_sample = (i + 1) * slice_length
                
                # 提取数据
                if channels == 1:
                    slice_data = audio_data[start_sample:end_sample]
                else:
                    slice_data = audio_data[start_sample:end_sample, :]
                
                # 生成文件名
                base_name = os.path.splitext(jpg_file)[0]
                output_filename = f"{base_name}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存切片
                with wave.open(output_path, 'wb') as output_wav:
                    output_wav.setnchannels(channels)
                    output_wav.setsampwidth(sample_width)
                    output_wav.setframerate(sample_rate)
                    output_wav.writeframes(slice_data.tobytes())
            
            return True
            
    except Exception as e:
        print(f"处理 {os.path.basename(wav_path)} 时出错: {e}")
        return False

def process_directory(target_dir):
    """处理指定目录下的所有WAV文件"""
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"错误：目录不存在 - {target_dir}")
        return
    
    # 获取所有WAV文件
    wav_files = list(target_path.glob("*.wav"))
    
    if not wav_files:
        print(f"在目录 {target_dir} 中未找到WAV文件")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for wav_file in wav_files:
        # 获取文件名（不含扩展名）
        wav_name = wav_file.stem
        
        # 寻找对应的JPG目录
        jpg_dir = target_path / wav_name
        
        if not jpg_dir.exists():
            print(f"跳过 {wav_file.name} - 未找到对应目录 {wav_name}")
            skipped_count += 1
            continue
        
        # 获取JPG文件
        jpg_files = get_jpg_files(jpg_dir)
        
        if not jpg_files:
            print(f"跳过 {wav_file.name} - 目录 {wav_name} 中无JPG文件")
            skipped_count += 1
            continue
        
        # 创建输出目录
        output_dir = target_path / f"{wav_name}_wav"
        
        # 执行切片
        if slice_wav_file(str(wav_file), str(output_dir), jpg_files):
            print(f"完成 {wav_file.name} → {wav_name}_wav/ ({len(jpg_files)} 个切片)")
            processed_count += 1
        else:
            print(f"失败 {wav_file.name}")
            skipped_count += 1
    
    print(f"\n处理完成: {processed_count} 个成功, {skipped_count} 个跳过")

def main():
    parser = argparse.ArgumentParser(description='自动处理目录中的WAV文件切片')
    parser.add_argument('directory', nargs='?', default='.', 
                       help='要处理的目录路径（默认为当前目录）')
    
    args = parser.parse_args()
    
    print(f"处理目录: {os.path.abspath(args.directory)}")
    print("-" * 50)
    
    process_directory(args.directory)

if __name__ == "__main__":
    main()
