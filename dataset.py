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
            label_frames = []

            input_dir = video
            # input_dir = self.add_skeleton_suffix(video, "process")
            input_dir_skeletion = self.add_suffix_to_parent(video, "skeleton", 1)
            frame_list = glob.glob(os.path.join(input_dir, '*.jpg'))
            frame_list_skeletion = glob.glob(os.path.join(input_dir_skeletion, '*.jpg'))
            # frame_list.sort()
            # frame_list_skeletion.sort()

            for t in range(T, T + self.args.sample_frames):
            # for t in range(0, N):
                input_frame = cv2.imread(frame_list[t])
                input_frame_skeletion = cv2.imread(frame_list_skeletion[t])
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_frame_skeletion = cv2.cvtColor(input_frame_skeletion, cv2.COLOR_BGR2RGB)
                input_frame = Image.fromarray(input_frame)
                input_frame_skeletion = Image.fromarray(input_frame_skeletion)

                input_frames.append(self.transforms(input_frame))
                input_frames_skeletion.append(self.transforms(input_frame_skeletion))

                name_without_ext = os.path.splitext(frame_list[t])[0]
                parts = name_without_ext.rsplit('_', 1)
                lookup_indices = self.labels.index(parts[-1])
                label_tensors = self.tensor_labels[lookup_indices]
                label_frames.append(label_tensors)

            input_frames = torch.stack(input_frames, 0)
            label_frames = torch.stack(label_frames, 0)
            input_frames_skeletion = torch.stack(input_frames_skeletion, 0)

            return torch.cat([input_frames, input_frames_skeletion], dim=1), label_frames

        else:
            N = self.num_frames[index]
            # T = random.randint(0, N - self.args.sample_frames_temp)
            video = self.input_file[index]
            input_frames = []
            input_frames_skeletion = []
            label_frames = []

            input_dir = video
            # input_dir = self.add_skeleton_suffix(video, "process")
            input_dir_skeletion = self.add_skeleton_suffix(video, "skeleton")
            frame_list = glob.glob(os.path.join(input_dir, '*.jpg'))
            frame_list_skeletion = glob.glob(os.path.join(input_dir_skeletion, '*.jpg'))
            frame_list.sort()
            frame_list_skeletion.sort()

            # for t in range(T, T + self.args.sample_frames_temp):
            for t in range(1860, 1960):
            # for t in range(0, 100):
                input_frame = cv2.imread(frame_list[t])
                input_frame_skeletion = cv2.imread(frame_list_skeletion[t])
                # print(frame_list_skeletion[t])
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_frame_skeletion = cv2.cvtColor(input_frame_skeletion, cv2.COLOR_BGR2RGB)
                input_frame = Image.fromarray(input_frame)
                input_frame_skeletion = Image.fromarray(input_frame_skeletion)

                input_frames.append(self.transforms(input_frame))
                input_frames_skeletion.append(self.transforms(input_frame_skeletion))

                name_without_ext = os.path.splitext(frame_list[t])[0]
                parts = name_without_ext.rsplit('_', 1)
                lookup_indices = self.labels.index(parts[-1])
                label_tensors = self.tensor_labels[lookup_indices]
                label_frames.append(label_tensors)

            input_frames = torch.stack(input_frames, 0)
            label_frames = torch.stack(label_frames, 0)
            input_frames_skeletion = torch.stack(input_frames_skeletion, 0)

            return torch.cat([input_frames, input_frames_skeletion], dim=1), label_frames

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_file)
