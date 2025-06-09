import os
import cv2
import torch
import glob
import torchvision
import torch.utils.data
import PIL
import re
import random
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import librosa
import numpy as np


class TIC:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.transforms = torchvision.transforms.Compose([transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True):
        print("=> evaluating Tic data set...")
        train_dataset = TICDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'TIC'),
                                   transforms=self.transforms,
                                   isTest=False,
                                   parse_patches=parse_patches,
                                   config=self.config,
                                   args=self.args)
        val_dataset = TICDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'TIC'),
                                 transforms=self.transforms,
                                 isTest=True,
                                 parse_patches=parse_patches,
                                 config=self.config,
                                 args=self.args)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class TICDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transforms, isTest=False, parse_patches=True, config=None, args=None):
        super().__init__()
        self.num_frames = []
        self.args = args

        self.n_mfcc = getattr(args, 'n_mfcc', 13)  # MFCC特征维度
        self.audio_sr = getattr(args, 'audio_sr', 16000)  # 音频采样率
        self.max_audio_length = getattr(args, 'max_audio_length', 100)  # 最大音频序列长度

        if not isTest:
            train_list = dir + '/' + 'train_tic.txt'
            with open(train_list) as f:
                contents = f.readlines()
                input_file = [dir + '/' + 'train/' + i.rstrip("\n") for i in contents]

            for video in input_file:
                frame_list = glob.glob(os.path.join(video, '*.jpg'))
                frame_list.sort()
                if len(frame_list) == 0:
                    raise Exception("No frames in %s" % video)

                self.num_frames.append(len(frame_list))

        else:
            test_list = dir + '/' + 'test_tic.txt'
            with open(test_list) as f:
                contents = f.readlines()
                input_file = [dir + '/' + 'test/' + i.rstrip("\n") for i in contents]

            for video in input_file:
                frame_list = glob.glob(os.path.join(video, '*.jpg'))
                frame_list.sort()
                if len(frame_list) == 0:
                    raise Exception("No frames in %s" % video)

                self.num_frames.append(len(frame_list))

        self.input_file = input_file
        self.dir = dir
        self.transforms = transforms
        self.config = config
        self.parse_patches = parse_patches
        self.isTest = isTest
        self.labels = args.labels
        df = pd.DataFrame(self.labels, columns=['tic'])
        one_hot_encoded = pd.get_dummies(df['tic'])
        self.tensor_labels = torch.tensor(one_hot_encoded.values, dtype=torch.float32)

    def add_skeleton_suffix(self, original_path, change_name):
        """自动在路径最后一个目录名后添加_skeleton后缀"""
        # 规范化路径（统一分隔符）
        normalized_path = os.path.normpath(original_path)

        # 分割路径的目录和文件名部分
        dir_name = os.path.dirname(normalized_path)
        base_name = os.path.basename(normalized_path)

        # 添加_skeleton后缀
        new_base = f"{base_name}_{change_name}"

        # 重组路径
        new_path = os.path.join(dir_name, new_base)
        return new_path
    
    
    
    def get_audio_path(self, video_path):
        """根据视频路径获取对应的音频文件路径"""
        # 将视频目录转换为音频目录路径
        audio_path = video_path.replace('/train/', '/train_audio/').replace('/test/', '/test_audio/')
        # 查找音频文件（支持多种格式）
        for ext in ['.wav', '.mp3', '.m4a', '.flac']:
            audio_file = audio_path + ext
            if os.path.exists(audio_file):
                return audio_file
        return None

    def extract_mfcc_features(self, audio_path):
        """提取MFCC特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.audio_sr)
            
            # 如果音频太短，进行填充
            if len(y) < sr * 0.5:  # 小于0.5秒
                y = np.tile(y, int(sr * 0.5 / len(y)) + 1)[:int(sr * 0.5)]
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=1024, hop_length=256
            )
            
            # 添加差分特征
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 合并特征 (39维：13 MFCC + 13 Delta + 13 Delta2)
            features = np.concatenate([mfcc, delta, delta2], axis=0)
            features = features.T  # 转置为 (时间, 特征)
            
            # 长度标准化
            if features.shape[0] > self.max_audio_length:
                features = features[:self.max_audio_length]
            else:
                # 填充到固定长度
                pad_length = self.max_audio_length - features.shape[0]
                features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
            
            return torch.tensor(features, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            # 返回零特征作为备用
            return torch.zeros((self.max_audio_length, self.n_mfcc * 3), dtype=torch.float32)


    def add_suffix_to_parent(self, original_path, suffix, depth=1):
        """
        在指定深度的父目录名后添加后缀

        参数:
        original_path: 原始路径
        suffix: 要添加的后缀
        depth: 要修改的父目录层级（从文件名开始计数，默认1表示直接父目录）
        """
        normalized_path = os.path.normpath(original_path)
        path_parts = normalized_path.split(os.sep)

        if depth >= len(path_parts):
            raise ValueError(f"路径深度不足: depth={depth}, path={normalized_path}")

        # 修改指定层级的目录名
        target_index = len(path_parts) - depth - 1
        path_parts[target_index] = f"{path_parts[target_index]}_{suffix}"

        return os.sep.join(path_parts)


    def get_images(self, index):
        if not self.isTest:
            N = self.num_frames[index]
            T = random.randint(0, N - self.args.sample_frames)
            video = self.input_file[index]
            input_frames = []
            input_frames_skeletion = []
            input_frames_audio = []
            label_frames = []

            input_dir = video
            # input_dir = self.add_skeleton_suffix(video, "process")
            input_dir_skeletion = self.add_suffix_to_parent(video, "skeleton", 1)
            input_dir_audio = self.add_suffix_to_parent(video, "wav", 1)
            frame_list = glob.glob(os.path.join(input_dir, '*.jpg'))
            frame_list_skeletion = glob.glob(os.path.join(input_dir_skeletion, '*.jpg'))
            frame_list_audio = glob.glob(os.path.join(input_dir_audio, '*.wav'))
            if(len(frame_list_audio) != len(frame_list)):
                print(f"错误: {video} 的音频文件数量与图像文件数量不匹配")
            # frame_list.sort()
            # frame_list_skeletion.sort()

            for t in range(T, T + self.args.sample_frames):
            # for t in range(0, N):
                input_frame = cv2.imread(frame_list[t])
                input_frame_skeletion = cv2.imread(frame_list_skeletion[t])
                input_frame_audio = self.extract_mfcc_features(frame_list_audio[t])
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_frame_skeletion = cv2.cvtColor(input_frame_skeletion, cv2.COLOR_BGR2RGB)
                input_frame = Image.fromarray(input_frame)
                input_frame_skeletion = Image.fromarray(input_frame_skeletion)

                input_frames.append(self.transforms(input_frame))
                input_frames_skeletion.append(self.transforms(input_frame_skeletion))
                input_frames_audio.append(input_frame_audio)

                name_without_ext = os.path.splitext(frame_list[t])[0]
                parts = name_without_ext.rsplit('_', 1)
                lookup_indices = self.labels.index(parts[-1])
                label_tensors = self.tensor_labels[lookup_indices]
                label_frames.append(label_tensors)

            input_frames = torch.stack(input_frames, 0)
            label_frames = torch.stack(label_frames, 0)
            input_frames_skeletion = torch.stack(input_frames_skeletion, 0)
            input_frames_audio = torch.stack(input_frames_audio, 0)
            return torch.cat([input_frames, input_frames_skeletion], dim=1), input_frames_audio, label_frames

        else:
            N = self.num_frames[index]
            # T = random.randint(0, N - self.args.sample_frames_temp)
            video = self.input_file[index]
            input_frames = []
            input_frames_skeletion = []
            input_frames_audio = []
            label_frames = []

            input_dir = video
            # input_dir = self.add_skeleton_suffix(video, "process")
            input_dir_skeletion = self.add_skeleton_suffix(video, "skeleton")
            input_dir_audio = self.add_skeleton_suffix(video, "wav")
            print(input_dir_audio)
            frame_list = glob.glob(os.path.join(input_dir, '*.jpg'))
            frame_list_skeletion = glob.glob(os.path.join(input_dir_skeletion, '*.jpg'))
            frame_list_audio = glob.glob(os.path.join(input_dir_audio, '*.wav'))
            frame_list.sort()
            frame_list_skeletion.sort()

            # for t in range(T, T + self.args.sample_frames_temp):
            for t in range(1860, 1960):
            # for t in range(0, 100):
                input_frame = cv2.imread(frame_list[t])
                input_frame_skeletion = cv2.imread(frame_list_skeletion[t])
                input_frame_audio = self.extract_mfcc_features(frame_list_audio[t])
                # print(frame_list_skeletion[t])
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_frame_skeletion = cv2.cvtColor(input_frame_skeletion, cv2.COLOR_BGR2RGB)
                input_frame = Image.fromarray(input_frame)
                input_frame_skeletion = Image.fromarray(input_frame_skeletion)

                input_frames.append(self.transforms(input_frame))
                input_frames_skeletion.append(self.transforms(input_frame_skeletion))
                input_frames_audio.append(input_frame_audio)

                name_without_ext = os.path.splitext(frame_list[t])[0]
                parts = name_without_ext.rsplit('_', 1)
                lookup_indices = self.labels.index(parts[-1])
                label_tensors = self.tensor_labels[lookup_indices]
                label_frames.append(label_tensors)

            input_frames = torch.stack(input_frames, 0)
            label_frames = torch.stack(label_frames, 0)
            input_frames_skeletion = torch.stack(input_frames_skeletion, 0)
            input_frames_audio = torch.stack(input_frames_audio, 0)
            return torch.cat([input_frames, input_frames_skeletion], dim=1), input_frames_audio, label_frames

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_file)
