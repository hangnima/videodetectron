import time

from network import TicLoss, TicDetector, VideoTransformer
from network2 import VideoTransformer2
from network3 import MultiModalTicDetector
from dataset import TIC
import argparse
import os
import yaml
import torch
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training TIC Models')
    parser.add_argument("--config", default='/data/hym/tic3/videodetectron/tic.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('-learning_rate', help='Set the learning rate', default=0.0001, type=float)
    parser.add_argument('-learning_rate_max', help='Set the learning rate', default=0.0002, type=float)
    parser.add_argument("--device_ids", type=list, default=[0])
    parser.add_argument('--sample_frames', help='number of video frames', default=5, type=int)
    parser.add_argument('--window_size', help='window_size', default=5, type=int)
    parser.add_argument('-save_pic', help='the place to save pic', default='./out_pic/',
                        type=str)
    parser.add_argument('-save_ckp', help='the place to save pic', default='./ckp/',
                        type=str)
    parser.add_argument('-is_train', help='train or test', default=True,
                        type=bool)
    parser.add_argument('--labels', type=list, default=['None', 'face-tic', 'head-tic', 'body-tic', 'vocal-tic'], help='text label')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def adjust_learning_rate(optimizer, epoch, num_epochs, warmup_lr, max_lr):

    warmup_epochs = 30
    max_epochs = num_epochs

    if epoch <= warmup_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = ((max_lr - warmup_lr) / warmup_epochs) * epoch + warmup_lr
            # print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = max_lr - ((max_lr - warmup_lr) / (max_epochs - warmup_epochs)) * (epoch - warmup_epochs)
            # print('Learning rate sets to {}.'.format(param_group['lr']))

def main():
    args, config = parse_args_and_config()
    tic_data = TIC(config, args)
    train_loader, val_loader = tic_data.get_loaders()
    num_epochs = 100

    # --- Gpu device --- #
    # device_ids = [Id for Id in range(torch.cuda.device_count())]
    device_ids = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.is_train==True:
        # ---define network---
        net = MultiModalTicDetector(args)
        #net = TicDetector(args)
        '''
        net = VideoTransformer2(args,num_classes=5,
                 img_size=224,
                 patch_size=16,
                 num_frames=args.sample_frames,
                 dim=768,
                 depth=12,
                 heads=8)
        '''
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)
        #ckpt = torch.load(f'{args.save_ckp}/ckp_199')
        #net.load_state_dict((ckpt['net']), strict=True)
        epoch = 0
        #epoch = ckpt['epoch']

        # ---data load---
        data_loader = TIC(config,args)
        train_loader, val_loader = data_loader.get_loaders()

        # --- Build optimizer --- #
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
        while epoch < num_epochs:
            psnr_list = []
            sum_loss = 0
            jisu=1

            adjust_learning_rate(optimizer, epoch, num_epochs, args.learning_rate, args.learning_rate_max)
            start=time.time()
            for i, (data, audio, label) in enumerate(train_loader):
                input_frame, input_frame_skeleton = data[:, :, :3, :], data[:, :, 3:, :]
                input_frame = input_frame.to(device)
                input_frame_skeleton = input_frame_skeleton.to(device)
                input_frame_audio = audio.to(device)
                #print(input_frame_audio.shape)
                #print(input_frame.shape)
                # label = label[:,args.window_size:-args.window_size,:].to(device)
                label = label.to(device)
                B, T, C, H, W = input_frame.shape
                label_len = label.size(2)
                label_flat = torch.argmax(label.view(-1, label_len), dim=1)

                # --- Zero the parameter gradients --- #
                optimizer.zero_grad()

                # --- Forward + Backward + Optimize --- #
                net.train()
                classifer = net(input_frame.permute(0, 2, 1, 3, 4), input_frame_skeleton.permute(0, 2, 1, 3, 4),
                                input_frame_audio)
                # classifer = net(input_frame, input_frame_skeleton)
                # --- Calculate Total loss --- #
                total_loss = criterion(classifer.view(-1,label_len), label_flat)
                sum_loss+=total_loss
                jisu+=1
                total_loss.backward()
                optimizer.step()

                # if not (i % 1000):
                #     print('Epoch: {0}, Iteration: {1}'.format(epoch, i))
                #     print('total_loss = {0}'.format(total_loss))
            end = time.time()
            print(f"one epoch uses time {end-start}")
            if not (epoch % 10):
                print('Epoch: {0}'.format(epoch))
                print('total_loss = {0}'.format(sum_loss/jisu))
            # net.eval()

            # if epoch % 20 == 0 or epoch == num_epochs-1:
            if epoch == num_epochs - 1 or epoch % 100 == 0:
                state = {'net': net.state_dict(), 'epoch': epoch}
                os.makedirs(os.path.join(args.save_ckp), exist_ok=True)
                torch.save(state, f'{args.save_ckp}/ckp_{epoch}')

            # if epoch == num_epochs - 1 or epoch % 100 == 0:
            #     val_psnr, val_ssim = validation(net, val_data_loader, device, args)
            #     print_log(epoch + 1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)

            epoch += 1
    else:
        # ---define network---
        # net = TicDetector(args)
        '''
        net = VideoTransformer2(args, num_classes=5,
                               img_size=224,
                               patch_size=16,
                               num_frames=args.sample_frames,
                               dim=768,
                               depth=12,
                               heads=8)
        '''
        net = MultiModalTicDetector(args)
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)
        ckpt = torch.load(f'{args.save_ckp}/ckp_99')
        net.load_state_dict((ckpt['net']), strict=True)
        net.eval()

        # ---data load---
        data_loader = TIC(config, args)
        _, val_loader = data_loader.get_loaders()

        for i, (data, audio, label) in enumerate(val_loader):
            input_frame, input_frame_skeleton = data[:, :, :3, :], data[:, :, 3:, :]
            input_frame = input_frame.to(device)
            input_frame_skeleton = input_frame_skeleton.to(device)
            input_frame_audio = audio.to(device)
            label = label.to(device)
            label_len = label.size(2)
            label_flat = torch.argmax(label.view(-1, label_len), dim=1)
            B, T, C, H, W = input_frame.shape

            # --- Forward + Backward + Optimize --- #
            # classifer = net(input_frame, input_frame_skeleton)
            classifer = []
            for j in range(0, T//args.sample_frames):
                # 视频数据切片 [batch, time, channels, height, width]
                input_frame_temp = input_frame[:, j * args.sample_frames:(j + 1) * args.sample_frames, :, :, :]
                input_frame_skeleton_temp = input_frame_skeleton[:, j * args.sample_frames:(j + 1) * args.sample_frames, :, :, :]
                
                # 音频数据切片 [batch, time, audio_length, features] - 只有4维
                input_frame_audio_temp = input_frame_audio[:, j * args.sample_frames:(j + 1) * args.sample_frames, :, :]
                
                # 调用模型
                classifer.append(net(
                    input_frame_temp.permute(0, 2, 1, 3, 4), 
                    input_frame_skeleton_temp.permute(0, 2, 1, 3, 4), 
                    input_frame_audio_temp
                ))

                # classifer.append(net(input_frame_temp, input_frame_skeleton_temp))

            classifer = torch.stack(classifer,dim=0)
            classifer = torch.argmax(classifer.view(-1, label_len), dim=1)
            correct = (classifer == label_flat).sum().item()
            # error = (classifer == label_flat)
            # print(error)
            accuracy = 100.0 * correct / label_flat.size(0)  # 75.0%
            print(f"Accuracy: {accuracy:.2f}%")
            print(classifer)
            print(label_flat)

            # classifer = net(input_frame_skeleton.permute(0, 2, 1, 3, 4))
            # classifer = torch.argmax(classifer.view(-1, label_len), dim=1)
            #
            # correct = (classifer == label_flat).sum().item()
            # accuracy = 100.0 * correct / label_flat.size(0)  # 75.0%
            # print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()