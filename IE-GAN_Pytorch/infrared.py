'''
Author: your name
Date: 2021-03-24 00:30:19
LastEditTime: 2021-03-24 01:08:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /zlj/Infraredechance/torch/IE-GANbyzlj/infrared.py
'''
import os
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import h5py
import operator
import cv2
# from skimage.measure import compare_psnr

class infraredData(Dataset):#文件夹
    def __init__(self,img_dir,label_dir,transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir        
        self.imgs = os.listdir(self.img_dir)
        self.labels = os.listdir(self.label_dir)
        if transform == None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        img_index = self.imgs[index]
        label_index = self.labels[index]
        img_path = os.path.join(self.img_dir, img_index)
        label_path = os.path.join(self.label_dir, label_index)
        img = Image.open(img_path)
        label = Image.open(label_path) #RGB
        if self.transform:
            img = self.transform(img).float()#通道变换&归一化
            label= self.transform(label).float()
        return img,label

class infraredDatah5(Dataset):#h5格式 训练先将原始图像预先裁剪为256×256
    def __init__(self,source_h5_path,target_h5_path):
        self.source_h5 = h5py.File(source_h5_path, 'r')
        self.target_h5 = h5py.File(target_h5_path, 'r')
        assert operator.eq(list(self.source_h5.keys()), list(self.target_h5.keys()))
        self.keys = list(self.source_h5.keys())
    def __len__(self):
        return len(self.keys)
    def __getitem__(self,index):
        key = self.keys[index]
        source = np.array(self.source_h5[key])
        target = np.array(self.target_h5[key])
        source = torch.from_numpy(np.float32(source / 255)).permute(2, 0, 1)
        target = torch.from_numpy(np.float32(target / 255)).permute(2, 0, 1)
        return source,target

def image2patches(img, win, stride):  # 把图像转换位图像块（滑窗法）
    h, w, _ = img.shape
    assert win < h and win < w
    patches = []
    for i in range(0, h - win + 1, stride):
        for j in range(0, w - win + 1, stride):
            # print(img)
            patch = img[i:i + win, j:j + win, :]
            # print(patch)
            patches.append(patch)
    # print(patches)
    return np.array(patches)

class infraredDatah5testcrop256(Dataset):#h5格式 测试由于原始图像411×640 411不能被2整除,需要将原始图像调整为410 640
    def __init__(self,source_h5_path,target_h5_path):
        self.source_h5 = h5py.File(source_h5_path, 'r')
        self.target_h5 = h5py.File(target_h5_path, 'r')
        assert operator.eq(list(self.source_h5.keys()), list(self.target_h5.keys()))
        self.keys = list(self.source_h5.keys())
    def __len__(self):
        return len(self.keys)
    def __getitem__(self,index):
        key = self.keys[index]
        source = np.array(self.source_h5[key])
        target = np.array(self.target_h5[key])

        source = image2patches(source,256,40)
        target = image2patches(target,256,40)

        source = torch.from_numpy(np.float32(source / 255)).permute(0,3, 1, 2)
        target = torch.from_numpy(np.float32(target / 255)).permute(0,3, 1, 2)
        return source,target

class infraredDatah5test(Dataset):#h5格式 测试由于原始图像411×640 411不能被2整除,需要将原始图像调整为410 640
    def __init__(self,source_h5_path,target_h5_path):
        self.source_h5 = h5py.File(source_h5_path, 'r')
        self.target_h5 = h5py.File(target_h5_path, 'r')
        assert operator.eq(list(self.source_h5.keys()), list(self.target_h5.keys()))
        self.keys = list(self.source_h5.keys())
    def __len__(self):
        return len(self.keys)
    def __getitem__(self,index):
        key = self.keys[index]
        source = np.array(self.source_h5[key])
        target = np.array(self.target_h5[key])

        source = cv2.resize(source,(640,410))
        target = cv2.resize(target,(640,410))

        source = torch.from_numpy(np.float32(source / 255)).permute(2, 0, 1)
        target = torch.from_numpy(np.float32(target / 255)).permute(2, 0, 1)
        return source,target

if __name__ == '__main__':
    img_train_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_train_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    img_test_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_test_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    trainset = infraredDatah5(img_train_h5_path,label_train_h5_path)
    testset = infraredDatah5test(img_test_h5_path,label_test_h5_path)
    print(trainset.__len__(),testset.__len__())

    dataloader_train = DataLoader(trainset,32)
    dataloader_test = DataLoader(testset,1)

    testset.__getitem__(0)
    # for step,(x,y) in enumerate(dataloader_train):
    #     print(x.shape)#32,3,256,256
    #     print(y.shape)
    for step,(x,y) in enumerate(dataloader_test):
        print(x.shape)#1,3,256,256
        print(y.shape)
    # for step,(x,y) in enumerate(trainset):
    #     print(x.shape)#3,256,256
    #     print(y.shape)
    # for step,(x,y) in enumerate(testset):
    #     print(x.shape)#3,256,256
    #     print(y.shape)
    