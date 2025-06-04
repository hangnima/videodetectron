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

        # # 添加缓存机制
        # self.frame_cache = {}  # 缓存视频帧信息
        # self.image_cache = joblib.Memory(location='./image_cache', verbose=0)  # 磁盘缓存
        #
        # # 预加载所有视频的帧信息
        # self._precache_frame_info()

    # def _precache_frame_info(self):
    #     """预加载所有视频的帧信息到缓存"""
    #     print("预加载视频帧信息...")
    #     start_time = time.time()
    #
    #     for idx, video in enumerate(self.input_file):
    #         input_dir = video
    #         input_dir_skeletion = self.add_suffix_to_parent(video, "skeleton", 1)
    #
    #         # 获取并排序帧列表
    #         frame_list = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    #         skeleton_list = sorted(glob.glob(os.path.join(input_dir_skeletion, '*.jpg')))
    #
    #         # 缓存帧列表
    #         self.frame_cache[video] = {
    #             'frame_list': frame_list,
    #             'skeleton_list': skeleton_list,
    #             'frame_count': len(frame_list)
    #         }
    #
    #         labels = []
    #         for frame_path in frame_list:
    #             name_without_ext = os.path.splitext(frame_path)[0]
    #             parts = name_without_ext.rsplit('_', 1)
    #             lookup_indices = self.labels.index(parts[-1])
    #             labels.append(self.tensor_labels[lookup_indices])
    #         self.frame_cache[video]['labels'] = labels
    #
    #     print(f"帧信息预加载完成，耗时: {time.time() - start_time:.2f}秒")
    #
    # def _load_image_pair(self, frame_path, skeleton_path):
    #     """优化图像加载函数"""
    #     # 尝试从缓存加载
    #     cache_key = f"{frame_path}_{skeleton_path}"
    #     cached = self.image_cache.cache(cache_key)
    #     if cached is not None:
    #         return cached
    #
    #     # 直接使用PIL加载RGB图像，避免OpenCV转换
    #     frame_img = Image.open(frame_path).convert('RGB')
    #     skeleton_img = Image.open(skeleton_path).convert('RGB')
    #
    #     # 应用转换
    #     frame_tensor = self.transforms(frame_img)
    #     skeleton_tensor = self.transforms(skeleton_img)
    #
    #     # 缓存结果
    #     self.image_cache(cache_key, (frame_tensor, skeleton_tensor))
    #     return frame_tensor, skeleton_tensor

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

    # def get_images(self, index):
    #     # 从缓存获取视频信息
    #     video = self.input_file[index]
    #     video_info = self.frame_cache.get(video)
    #     # if not video_info:
    #     #     # 如果缓存中没有，实时加载（应该不会发生）
    #     #     input_dir = video
    #     #     input_dir_skeletion = self.add_suffix_to_parent(video, "skeleton", 1)
    #     #     frame_list = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    #     #     skeleton_list = sorted(glob.glob(os.path.join(input_dir_skeletion, '*.jpg')))
    #     #     video_info = {
    #     #         'frame_list': frame_list,
    #     #         'skeleton_list': skeleton_list,
    #     #         'frame_count': len(frame_list)
    #     #     }
    #     #     self.frame_cache[video] = video_info
    #
    #     frame_list = video_info['frame_list']
    #     skeleton_list = video_info['skeleton_list']
    #     frame_count = video_info['frame_count']
    #
    #     # 确定采样范围
    #     if not self.isTest:
    #         # 训练模式随机采样
    #         T = random.randint(0, max(0, frame_count - self.args.sample_frames))
    #         end = min(T + self.args.sample_frames, frame_count)
    #         indices = list(range(T, end))
    #     else:
    #         # 测试模式固定范围
    #         start = 1860
    #         end = min(1960, frame_count)
    #         indices = list(range(start, end))
    #
    #     # 并行加载图像对
    #     input_frames = []
    #     input_frames_skeletion = []
    #     label_frames = []
    #
    #     # 使用线程池并行加载
    #     with ThreadPoolExecutor(max_workers=min(1, os.cpu_count())) as executor:
    #         futures = []
    #         for i in indices:
    #             frame_path = frame_list[i]
    #             skeleton_path = skeleton_list[i]
    #             futures.append(executor.submit(self._load_image_pair, frame_path, skeleton_path))
    #
    #         for future in futures:
    #             try:
    #                 frame_tensor, skeleton_tensor = future.result()
    #                 input_frames.append(frame_tensor)
    #                 input_frames_skeletion.append(skeleton_tensor)
    #
    #                 # 处理标签
    #                 name_without_ext = os.path.splitext(frame_path)[0]
    #                 parts = name_without_ext.rsplit('_', 1)
    #                 lookup_indices = self.labels.index(parts[-1])
    #                 label_frames.append(self.tensor_labels[lookup_indices])
    #             except Exception as e:
    #                 print(f"加载图像失败: {e}")
    #
    #     # 如果测试模式且已预加载标签
    #     # if self.isTest and 'labels' in video_info:
    #     label_frames = [video_info['labels'][i] for i in indices]
    #
    #     # 转换为张量
    #     input_frames = torch.stack(input_frames, 0) if input_frames else torch.tensor([])
    #     input_frames_skeletion = torch.stack(input_frames_skeletion, 0) if input_frames_skeletion else torch.tensor([])
    #     label_frames = torch.stack(label_frames, 0) if label_frames else torch.tensor([])
    #
    #     return torch.cat([input_frames, input_frames_skeletion], dim=1), label_frames

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
