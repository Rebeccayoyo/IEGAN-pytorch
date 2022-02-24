'''
Author: your name
Date: 2021-01-22 21:07:45
LastEditTime: 2021-03-24 00:47:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /红外图像增强/torch/IE-GANbyzlj/IE-GAN.py
'''
import os
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchvision import transforms
from torch.utils.tensorboard  import SummaryWriter
from infrared import infraredData,infraredDatah5,infraredDatah5test,infraredDatah5testcrop256
from PIL import Image
import glob
from cv2 import cv2
import numpy as np
import math
# from skimage.measure.simple_metrics import compare_psnr
##########相关结构图见笔记本端项目 红外增强结构图.word
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi_cuda = True
cuda = True

def cal_psnr(predict,label,data_range):#1or255
    predict = np.float64(predict) / data_range
    label = np.float64(label) / data_range
    mse = np.mean(np.square(predict-label))
    if mse ==0:
        return 100
    else:
        PIXEL_MAX = 1
        return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += cal_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

class IE_GPart(nn.Module):#encode and decode
    def __init__(self):
        super(IE_GPart,self).__init__()
        self.G_conv1 = nn.Conv2d(3,64,4,2,1)
        self.G_conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn2 = nn.BatchNorm2d(64)
        self.G_conv3 = nn.Conv2d(64,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.G_deconv4 = nn.ConvTranspose2d(128,3,4,2,1)
        self.G_relu = nn.ReLU()
        self.G_tanh = nn.Tanh()
    def forward(self,x):        
        x = self.G_conv1(x)
        x = self.G_relu(x)
        res = x 
        x = self.G_conv2(x)
        x = self.bn2(x)
        x = self.G_relu(x)

        x = self.G_conv3(x)
        x = self.bn3(x)
        x = self.G_relu(x)

        x = torch.cat((res,x),1)
        x = self.G_deconv4(x)        
        x = self.G_tanh(x)

        G_out = x
        return G_out

class IE_DPart(nn.Module):#判别网络
    def __init__(self):
        super(IE_DPart,self).__init__()
        self.D_conv1 = nn.Conv2d(6,64,4,2,1)
        self.D_conv2 = nn.Conv2d(64,128,4,2,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.D_conv3 = nn.Conv2d(128,256,4,2,1)
        self.bn3 = nn.BatchNorm2d(256)
        self.D_conv4 = nn.Conv2d(256,512,4,1,1)
        self.bn4 = nn.BatchNorm2d(512)
        self.D_conv5 = nn.Conv2d(512,1,4,1,1)
        self.D_leakyrelu = nn.LeakyReLU()
        self.fc = nn.Linear(900,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,input):
        # x = torch.cat([input,label],1)
        x = input 
        x = self.D_conv1(x)
        x = self.D_leakyrelu(x)
        
        x = self.D_conv2(x)
        x = self.bn2(x)
        x = self.D_leakyrelu(x)

        x = self.D_conv3(x)
        x = self.bn3(x)
        x = self.D_leakyrelu(x)

        x = self.D_conv4(x)
        x = self.bn4(x)
        x = self.D_leakyrelu(x)
         
        x = self.D_conv5(x)#26,26        
        x = x.view(x.size(0),-1)   
        x = self.fc(x)
        x = self.sigmoid(x)
        D_out = x
        return D_out

class AE_(nn.Module):#encode and decode 去掉res残差
    def __init__(self):
        super(AE_,self).__init__()
        self.G_conv1 = nn.Conv2d(3,64,4,2,1)
        self.G_conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn2 = nn.BatchNorm2d(64)
        self.G_conv3 = nn.Conv2d(64,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.G_deconv4 = nn.ConvTranspose2d(64,3,4,2,1)#128-64
        self.G_relu = nn.ReLU()
        self.G_tanh = nn.Tanh()
    def forward(self,x):        
        x = self.G_conv1(x)
        x = self.G_relu(x)
        res = x 
        x = self.G_conv2(x)
        x = self.bn2(x)
        x = self.G_relu(x)

        x = self.G_conv3(x)
        x = self.bn3(x)
        x = self.G_relu(x)

        # x = torch.cat((res,x),1)
        x = self.G_deconv4(x)        
        x = self.G_tanh(x)

        G_out = x
        return G_out

class AE_zljplan1(nn.Module):#encode and decode 去掉res残差
    def __init__(self):
        super(AE_zljplan1,self).__init__()
        self.G_conv1 = nn.Conv2d(3,64,4,2,1)
        self.G_conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn2 = nn.BatchNorm2d(64)
        self.G_conv3 = nn.Conv2d(64,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.G_deconv4 = nn.ConvTranspose2d(128,64,4,2,1)#128-64
        self.G_conv5 = nn.Conv2d(64,32,3,1,1)
        self.bn5 = nn.BatchNorm2d(32)
        self.G_conv6 = nn.Conv2d(32,3,3,1,1)
        self.G_relu = nn.ReLU()
        self.G_tanh = nn.Tanh()
    def forward(self,x):        
        x = self.G_conv1(x)
        x = self.G_relu(x)
        res = x 
        x = self.G_conv2(x)
        x = self.bn2(x)
        x = self.G_relu(x)

        x = self.G_conv3(x)
        x = self.bn3(x)
        x = self.G_relu(x)

        x = torch.cat((res,x),1)#128
        x = self.G_deconv4(x)        #64
        x = self.G_relu(x)

        x = self.G_conv5(x)#32
        x = self.bn5(x)
        x = self.G_relu(x)

        x = self.G_conv6(x)#3
        x = self.G_tanh(x)

        G_out = x
        return G_out

class AE_zljplan2(nn.Module):#encode and decode 去掉res残差
    def __init__(self):
        super(AE_zljplan2,self).__init__()
        self.G_conv1 = nn.Conv2d(3,64,4,2,1)
        self.G_conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn2 = nn.BatchNorm2d(64)
        self.G_conv3 = nn.Conv2d(64,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)

        self.avp = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(64,64//4,1,1,0)
        self.fc2 = nn.Conv2d(64//4,64,1,1,0)
        self.sigmoid = nn.Sigmoid()

        self.G_deconv4 = nn.ConvTranspose2d(128,64,4,2,1)#128-64
        self.G_conv5 = nn.Conv2d(64,32,3,1,1)
        self.bn5 = nn.BatchNorm2d(32)
        self.G_conv6 = nn.Conv2d(32,3,3,1,1)
        self.G_relu = nn.ReLU()
        self.G_tanh = nn.Tanh()
    def forward(self,x):        
        x = self.G_conv1(x)
        x = self.G_relu(x)
        res = x 
        x = self.G_conv2(x)
        x = self.bn2(x)
        x = self.G_relu(x)

        x = self.G_conv3(x)
        x = self.bn3(x)
        x = self.G_relu(x)

        se_x = x
        se_x = self.avp(se_x) 
        se_x = self.fc1(se_x)
        se_x = self.G_relu(se_x)
        se_x = self.fc2(se_x)
        se_x = self.sigmoid(se_x)
        x = se_x*x

        x = torch.cat((res,x),1)#128
        x = self.G_deconv4(x)        #64
        x = self.G_relu(x)

        x = self.G_conv5(x)#32
        x = self.bn5(x)
        x = self.G_relu(x)

        x = self.G_conv6(x)#3
        x = self.G_tanh(x)

        G_out = x
        return G_out

class AE_zljplan3(nn.Module):#encode and decode 去掉res残差
    def __init__(self):
        super(AE_zljplan3,self).__init__()
        self.G_conv1 = nn.Conv2d(3,64,4,2,1)
        self.G_conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn2 = nn.BatchNorm2d(64)
        self.G_conv3 = nn.Conv2d(64,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)

    #------------------通道注意力机制-----------------------------
        self.avp = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(64,64//4,1,1,0)
        self.fc2 = nn.Conv2d(64//4,64,1,1,0)
        self.sigmoid = nn.Sigmoid()
    #------------------空间注意力机制-----------------------------
        #avg=torch.mean(x,axis=1)
        #max=torch.max(x,axis=1)
        #torch.concat([avg,max],axis=1)
        self.loc_conv = nn.Conv2d(2,1,3,1,1)
        self.loc_activate = nn.Sigmoid()
        #* 对应矩阵 对应像素点乘

        self.G_deconv4 = nn.ConvTranspose2d(128,64,4,2,1)#128-64
        self.G_conv5 = nn.Conv2d(64,32,3,1,1)
        self.bn5 = nn.BatchNorm2d(32)
        self.G_conv6 = nn.Conv2d(32,3,3,1,1)
        self.G_relu = nn.ReLU()
        self.G_tanh = nn.Tanh()
    def forward(self,x):        
        x = self.G_conv1(x)
        x = self.G_relu(x)
        res = x 
        x = self.G_conv2(x)
        x = self.bn2(x)
        x = self.G_relu(x)

        x = self.G_conv3(x)
        x = self.bn3(x)
        x = self.G_relu(x)

        se_x = x#通道注意力
        se_x = self.avp(se_x) 
        se_x = self.fc1(se_x)
        se_x = self.G_relu(se_x)
        se_x = self.fc2(se_x)
        se_x = self.sigmoid(se_x)
        x = se_x*x

        loc_x = x#空间注意力
        loc_avg = torch.mean(loc_x,dim=1,keepdim=True)#batch,channel,width,height
        loc_max,_ = torch.max(loc_x,dim=1,keepdim=True)
        loc_all = torch.cat([loc_avg,loc_max],dim=1)
        # print(loc_all.shape)
        loc_all = self.loc_conv(loc_all)
        loc_all = self.loc_activate(x)
        loc_all = loc_all*loc_x

        x = loc_all

        x = torch.cat((res,x),1)#128
        x = self.G_deconv4(x)        #64
        x = self.G_relu(x)

        x = self.G_conv5(x)#32
        x = self.bn5(x)
        x = self.G_relu(x)

        x = self.G_conv6(x)#3
        x = self.G_tanh(x)

        G_out = x
        return G_out


def train_model_byimages():#以图像文件夹训练模型
    img_dir='/home/ipsg/code/ZLJ/Datasets/Infrared/2021323/LC_L/'
    label_dir = '/home/ipsg/code/ZLJ/Datasets/Infrared/2021323/LC_H/'
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),#维度变换 通道在前;以及归一化
    ])
    trainset = infraredData(img_dir,label_dir,transform)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)


    G_net = IE_GPart()
    D_net = IE_DPart()

    G_net.train()
    D_net.train()
    if cuda:
        G_net = G_net.cuda()
        D_net = D_net.cuda()
    if multi_cuda:
        G_net = torch.nn.DataParallel(G_net)
        D_net = torch.nn.DataParallel(D_net)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(G_net.parameters(),lr =lr)
    d_optimizer = torch.optim.Adam(D_net.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()
    criterionGAN = nn.BCELoss()
   
    EPOCH = 500
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################maxD#############################
            fake_image = G_net(batch_x)            
            fake_AB = torch.cat((batch_x,fake_image),1)#GAN_G
            fake_predict = D_net(fake_AB.detach()) #[fake_AB 是G网络的输出]tensor detach()该变量以及该变量以前的变量都不需要计算梯度 
            loss_gan_fake = criterionGAN(fake_predict,y_fake)        
            
            real_AB = torch.cat((batch_x,batch_y),1)#GAN_D
            real_predict = D_net(real_AB)
            loss_gan_real = criterionGAN(real_predict,y_real)

            loss_d = (loss_gan_fake+loss_gan_real)*0.5
            d_optimizer.zero_grad()
            loss_d.backward(retain_graph=True)           
            d_optimizer.step()
            #################minG#############################
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            fake_AB = torch.cat((batch_x,fake_image),1)#GAN_G
            fake_predict = D_net(fake_AB)
            loss_d = criterionGAN(fake_predict,y_real) # GAN_G 让D认为是真的
            
            loss_g = loss_L2loss*100 +loss_d
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()
            writer.add_scalar('dloss',loss_d.item(),iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'glloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if multi_cuda:
            torch.save(G_net.module.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_AE/G_Net' +
                        str(epoch) + '.pth')
        else:
            torch.save(G_net.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_AE/G_Net' +
                        str(epoch) + '.pth')

def train_modelnoD_byimages():#以图像文件夹训练模型
    img_dir='/home/ipsg/code/ZLJ/Datasets/Infrared/2021323/LC_L/'
    label_dir = '/home/ipsg/code/ZLJ/Datasets/Infrared/2021323/LC_H/'
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),#维度变换 通道在前;以及归一化
    ])
    trainset = infraredData(img_dir,label_dir,transform)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)


    G_net = IE_GPart()

    G_net.train()
    if cuda:
        G_net = G_net.cuda()
    if multi_cuda:
        G_net = torch.nn.DataParallel(G_net)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(G_net.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()

    
    EPOCH = 500
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            fake_image = G_net(batch_x)
            #################minG#############################
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            fake_predict = D_net(fake_AB) 
            loss_g = loss_L2loss*100#noD
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.add_scalar('gganloss',loss_gan_fake.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'glloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if multi_cuda:
            torch.save(G_net.module.state_dict(), '/home/ipsg/code/ZLJ/Infrared/torch_lua/IE-GAN_Pytorch_zlj/weights_IEGANnoD/G_Net' +
                        str(epoch) + '.pth')
        else:
            torch.save(G_net.state_dict(), '/home/ipsg/code/ZLJ/Infrared/torch_lua/IE-GAN_Pytorch_zlj/weights_IEGANnoD/G_Net' +
                        str(epoch) + '.pth')

def test_G_byimages():#获得图像
     G_net = IE_GPart()
     G_net.load_state_dict(torch.load('/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/imgfileresults/weights_IEGAN/G_Net149.pth'))
     G_net.eval()
     hig_img_dir ='/home/ipsg/code/ZLJ/Datasets/Infrared/2021324/LC_H/'
     test_img_dir = '/home/ipsg/code/ZLJ/Datasets/Infrared/2021324/LC_L/*.png'
     output_dir  = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/imgfileresults/results_IEGAN/LC22_color/output/'
     all_dir = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/imgfileresults/results_IEGAN/LC22_color/all/'
     if not os.path.exists(output_dir):
         os.makedirs(output_dir)
     if not os.path.exists(all_dir):
         os.makedirs(all_dir)
     files = glob.glob(test_img_dir)
     transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
    ])
     psnr_ = 0
     min_ = 100
     max_ = 0
     for f_ in files :
         img_name = f_.split('/')[-1]
         out_name = output_dir+img_name
         all_name = all_dir+img_name
         high_name = hig_img_dir + img_name

         img_= cv2.imread(f_)
         img_ = cv2.resize(img_,(256,256))
         image = Image.open(f_)
         image = transform(image)
         image=torch.unsqueeze(image,0)
         output = G_net.forward(image)

         tar = cv2.imread(high_name)
         tar = cv2.resize(tar,(256,256))

        #  output = torch.squeeze(output,0)
        #  output = transforms.ToPILImage()(output).convert('RGB')
        #  output.save(out_name)
         output = torch.squeeze(output,0).permute(1,2,0).detach().numpy()
         output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
         output = output*255

         all_img = np.concatenate([img_,tar,output],axis=1)
         cv2.imwrite(out_name,output)
         cv2.imwrite(all_name,all_img)
         r = cal_psnr(output,tar,255.)
         psnr_ += r
         if r>max_:max_ = r
         if r<min_:min_ = r
     print('num:',len(files),'psnr:',psnr_/len(files),min_,max_)

def test_GnoD_byimages():#获得图像
     G_net = IE_GPart()
     G_net.load_state_dict(torch.load('/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/imgfileresults/weights_IEGANnoD/G_Net149.pth'))
     G_net.eval()
     hig_img_dir ='/home/ipsg/code/ZLJ/Datasets/Infrared/2021324/LC_H/'
     test_img_dir = '/home/ipsg/code/ZLJ/Datasets/Infrared/2021324/LC_L/*.png'
     output_dir  = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/imgfileresults/results_IEGANnoD/LC22_color/output/'
     all_dir = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/imgfileresults/results_IEGANnoD/LC22_color/all/'
     if not os.path.exists(output_dir):
         os.makedirs(output_dir)
     if not os.path.exists(all_dir):
         os.makedirs(all_dir)
     files = glob.glob(test_img_dir)
     transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
    ])
     psnr_ = 0
     min_ = 100
     max_ = 0
     for f_ in files :
         img_name = f_.split('/')[-1]
         out_name = output_dir+img_name
         all_name = all_dir+img_name
         high_name = hig_img_dir + img_name

         img_= cv2.imread(f_)
         img_ = cv2.resize(img_,(256,256))
         image = Image.open(f_)
         image = transform(image)
         image=torch.unsqueeze(image,0)
         output = G_net.forward(image)

         tar = cv2.imread(high_name)
         tar = cv2.resize(tar,(256,256))

        #  output = torch.squeeze(output,0)
        #  output = transforms.ToPILImage()(output).convert('RGB')
        #  output.save(out_name)
         output = torch.squeeze(output,0).permute(1,2,0).detach().numpy()
         output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
         output = output*255

         all_img = np.concatenate([img_,tar,output],axis=1)
         cv2.imwrite(out_name,output)
         cv2.imwrite(all_name,all_img)
         r = cal_psnr(output,tar,255.)
         psnr_ += r
         if r>max_:max_ = r
         if r<min_:min_ = r
     print('num:',len(files),'psnr:',psnr_/len(files),min_,max_)

def train_model_byh5():#以h5格式训练模型
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=32,shuffle=True,num_workers=0)

    G_net = IE_GPart()
    D_net = IE_DPart()

    G_net.train()
    D_net.train()
    if cuda:
        G_net = G_net.cuda()
        D_net = D_net.cuda()
    if multi_cuda:
        G_net = torch.nn.DataParallel(G_net)
        D_net = torch.nn.DataParallel(D_net)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(G_net.parameters(),lr =lr)
    d_optimizer = torch.optim.Adam(D_net.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()
    criterionGAN = nn.BCELoss()

    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################最大值d#############################
            fake_image = G_net(batch_x)            
            fake_AB = torch.cat((batch_x,fake_image),1)#GAN_G
            fake_predict = D_net(fake_AB) 
            loss_gan_fake = criterionGAN(fake_predict,y_fake)        
            
            real_AB = torch.cat((batch_x,batch_y),1)#GAN_D
            real_predict = D_net(real_AB)
            loss_gan_real = criterionGAN(real_predict,y_real)

            loss_d = (loss_gan_fake+loss_gan_real)*0.5
            d_optimizer.zero_grad()
            loss_d.backward(retain_graph=True)           
            d_optimizer.step()
            #################minG#############################
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            fake_predict = D_net(fake_AB) 
            loss_gan_fake = criterionGAN(fake_predict,y_real)#GAN_D
            loss_g = loss_L2loss*100+loss_gan_fake#withD
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr = batch_PSNR(fake_image,batch_y,1)##没有变换通道应该不影响

            writer.add_scalar('psnr',psnr,iter)
            writer.add_scalar('dloss',loss_d.item(),iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.add_scalar('gganloss',loss_gan_fake.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'dloss:',loss_d.item(),'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item(),'gganloss:',loss_gan_fake.item())
        if epoch!=0 and epoch%30==0:
                if multi_cuda:
                    torch.save(G_net.module.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGAN' +
                                str(epoch) + '.pth')
                else:
                    torch.save(G_net.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGAN' +
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(G_net.module.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGAN' +
                    str(epoch) + '.pth')
    else:
        torch.save(G_net.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGAN' +
                    str(epoch) + '.pth')

def train_modelnoD_byh5(writer):#以h5格式训练模型
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)

    G_net = IE_GPart()
    G_net.train()
    if cuda:
        G_net = G_net.cuda()
    if multi_cuda:
        G_net = torch.nn.DataParallel(G_net)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(G_net.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()
   
    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################minG#############################
            fake_image = G_net(batch_x)     
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            loss_g = loss_L2loss*100#noD
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr = batch_PSNR(batch_y,fake_image,1)##没有变换通道应该不影响

            writer.add_scalar('psnr',psnr,iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.flush()


            print(epoch,'||',step,'||','lr:',lr,'psnr',psnr,'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if epoch!=0 and epoch%30==0:
                if multi_cuda:
                    torch.save(G_net.module.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGANnoD' +
                                str(epoch) + '.pth')
                else:
                    torch.save(G_net.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGANnoD' +
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(G_net.module.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGANnoD' +
                    str(epoch) + '.pth')
    else:
        torch.save(G_net.state_dict(), '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGANnoD' +
                    str(epoch) + '.pth')

def train_AE_byh5(writer):#以h5格式训练模型
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    weights_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/'+'weights_AE/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)

    AE = AE_()
    AE.train()
    if cuda:
        AE = AE.cuda()
    if multi_cuda:
        AE = torch.nn.DataParallel(AE)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(AE.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()

    
    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################minG#############################
            fake_image = AE(batch_x)
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            loss_g = loss_L2loss*100
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr_ = batch_PSNR(batch_y,fake_image,1.0)

            writer.add_scalar('psnr',psnr_,iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'psnr',psnr_,'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if epoch%30==0:
                if multi_cuda:
                    torch.save(AE.module.state_dict(), weigths_path +
                                str(epoch) + '.pth')
                else:
                    torch.save(AE.state_dict(), weigths_path +
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(AE.module.state_dict(), weigths_path +
                    str(epoch) + '.pth')
    else:
        torch.save(AE.state_dict(),weigths_path +
                    str(epoch) + '.pth')

def train_zlj_plan1(writer):
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    weights_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/'+'weights_zljplan1/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)

    AE = AE_zljplan1()
    AE.train()
    if cuda:
        AE = AE.cuda()
    if multi_cuda:
        AE = torch.nn.DataParallel(AE)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(AE.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()

    
    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################minG#############################
            fake_image = AE(batch_x)
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            loss_g = loss_L2loss*100
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr_ = batch_PSNR(batch_y,fake_image,1.0)

            writer.add_scalar('psnr',psnr_,iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'psnr',psnr_,'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if epoch%30==0:
                if multi_cuda:
                    torch.save(AE.module.state_dict(), weights_path +
                                str(epoch) + '.pth')
                else:
                    torch.save(AE.state_dict(), weights_path +
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(AE.module.state_dict(), weights_path +
                    str(epoch) + '.pth')
    else:
        torch.save(AE.state_dict(),weights_path +
                    str(epoch) + '.pth')

def train_zlj_plan2(writer):
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    weights_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/'+'weights_zljplan2/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)

    AE = AE_zljplan2()
    AE.train()
    if cuda:
        AE = AE.cuda()
    if multi_cuda:
        AE = torch.nn.DataParallel(AE)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(AE.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()

    
    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################minG#############################
            fake_image = AE(batch_x)
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            loss_g = loss_L2loss*100
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr_ = batch_PSNR(batch_y,fake_image,1.0)

            writer.add_scalar('psnr',psnr_,iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'psnr',psnr_,'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if epoch%30==0:
                if multi_cuda:
                    torch.save(AE.module.state_dict(), weights_path +
                                str(epoch) + '.pth')
                else:
                    torch.save(AE.state_dict(), weights_path +
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(AE.module.state_dict(), weights_path +
                    str(epoch) + '.pth')
    else:
        torch.save(AE.state_dict(),weights_path +
                    str(epoch) + '.pth')

def train_zlj_plan3(writer):
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    weights_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/'+'weights_zljplan3/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)

    AE = AE_zljplan3()
    AE.train()
    if cuda:
        AE = AE.cuda()
    if multi_cuda:
        AE = torch.nn.DataParallel(AE)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(AE.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()

    
    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################minG#############################
            fake_image = AE(batch_x)
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            loss_g = loss_L2loss*100
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr_ = batch_PSNR(batch_y,fake_image,1.0)

            writer.add_scalar('psnr',psnr_,iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'psnr',psnr_,'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item())
        if epoch%30==0:
                if multi_cuda:
                    torch.save(AE.module.state_dict(), weights_path +
                                str(epoch) + '.pth')
                else:
                    torch.save(AE.state_dict(), weights_path +
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(AE.module.state_dict(), weights_path +
                    str(epoch) + '.pth')
    else:
        torch.save(AE.state_dict(),weights_path +
                    str(epoch) + '.pth')

def train_zlj_plan4(writer):#以h5格式训练模型
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_train.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_train.h5'
    weights_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/'+'weights_zljplan4/'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    # transform = transforms.Compose([
    #     transforms.Resize([256,256]),
    #     transforms.ToTensor(),
    # ])
    trainset = infraredDatah5(img_h5_path,label_h5_path)
    data_loader = DataLoader(trainset,batch_size=32,shuffle=True,num_workers=0)

    G_net = AE_zljplan3()
    D_net = IE_DPart()

    G_net.train()
    D_net.train()
    if cuda:
        G_net = G_net.cuda()
        D_net = D_net.cuda()
    if multi_cuda:
        G_net = torch.nn.DataParallel(G_net)
        D_net = torch.nn.DataParallel(D_net)
    lr = 1e-3
    g_optimizer = torch.optim.Adam(G_net.parameters(),lr =lr)
    d_optimizer = torch.optim.Adam(D_net.parameters(),lr =lr)
    criterionL2 = nn.MSELoss()
    criterionGAN = nn.BCELoss()

    EPOCH = 100
    iter=0
    for epoch in range(EPOCH):
        if epoch != 0 and epoch%30 == 0:
            lr = lr*(0.1**(epoch//30))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = lr
        for step,(batch_x,batch_y) in enumerate(data_loader):
            iter += 1
            if cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            mini_batch = batch_x.size()[0]
            y_fake = torch.zeros(mini_batch)
            y_real = torch.ones(mini_batch)
            if cuda:
                y_fake,y_real = y_fake.cuda(),y_real.cuda()
            #################最大值d#############################
            fake_image = G_net(batch_x)            
            fake_AB = torch.cat((batch_x,fake_image),1)#GAN_G
            fake_predict = D_net(fake_AB) 
            loss_gan_fake = criterionGAN(fake_predict,y_fake)        
            
            real_AB = torch.cat((batch_x,batch_y),1)#GAN_D
            real_predict = D_net(real_AB)
            loss_gan_real = criterionGAN(real_predict,y_real)

            loss_d = (loss_gan_fake+loss_gan_real)*0.5
            d_optimizer.zero_grad()
            loss_d.backward(retain_graph=True)           
            d_optimizer.step()
            #################minG#############################
            loss_L2loss = criterionL2(fake_image,batch_y)#mse
            fake_predict = D_net(fake_AB) 
            loss_gan_fake = criterionGAN(fake_predict,y_real)#GAN_D
            loss_g = loss_L2loss*100+loss_gan_fake#withD
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

            psnr = batch_PSNR(batch_x,batch_y,1)##没有变换通道应该不影响

            writer.add_scalar('psnr',psnr,iter)
            writer.add_scalar('dloss',loss_d.item(),iter)
            writer.add_scalar('gl2loss',loss_L2loss.item(),iter)
            writer.add_scalar('gallloss',loss_g.item(),iter)
            writer.add_scalar('gganloss',loss_gan_fake.item(),iter)
            writer.flush()
            print(epoch,'||',step,'||','lr:',lr,'psnr:',psnr,'dloss:',loss_d.item(),'galloss:',loss_g.item(),'gl2loss:',loss_L2loss.item(),'gganloss:',loss_gan_fake.item())
        if epoch%30==0:
                if multi_cuda:
                    torch.save(G_net.module.state_dict(), weights_path +
                                str(epoch) + '.pth')
                else:
                    torch.save(G_net.state_dict(), weights_path+
                                str(epoch) + '.pth')
    if multi_cuda:
        torch.save(G_net.module.state_dict(), weights_path +
                    str(epoch) + '.pth')
    else:
        torch.save(G_net.state_dict(), weights_path +
                    str(epoch) + '.pth')

def test_G_byh5_crop256():#将测试集也crop成256成256大小
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_IEGAN/'    
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGAN/weights_IEGAN99.pth'
    G_net = IE_GPart()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5testcrop256(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        batches_x = batches_x.permute(1,0,2,3,4)
        batches_y = batches_y.permute(1,0,2,3,4)
        for index in range(batches_x.shape[0]):
            predict = G_net(batches_x[index])
            outname = output_path +str(step)+'.png'
            batch_x ,batch_y ,predict = torch.squeeze(batches_x[index],0).permute(1,2,0),torch.squeeze(batches_y[0],0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
            batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
            batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
            out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
            r = cal_psnr(predict,batch_y,255.)
            psnr_ += r
            if r < min_ :min_ = r
            if r > max_ :max_ = r
            cv2.imwrite(outname,out_all)
            count += 1
            print(r)
    print('num:',count,psnr_/count,min_,max_)

def test_G_byh5():#原图410×640测试
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_IEGAN/'    
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGAN/weights_IEGAN99.pth'
    G_net = IE_GPart()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5test(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        predict = G_net(batches_x)
        outname = output_path +str(step)+'.png'
        batch_x ,batch_y ,predict = torch.squeeze(batches_x,0).permute(1,2,0),torch.squeeze(batches_y,0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
        batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
        batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
        out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
        r = cal_psnr(predict,batch_y,255.)
        psnr_ += r
        if r < min_ :min_ = r
        if r > max_ :max_ = r
        cv2.imwrite(outname,out_all)
        count += 1
        print(r)
    print('num:',count,psnr_/count,min_,max_)

def test_GnoD_byh5():#原图410×640测试
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_IEGANnoD/'    
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_IEGANnoD/weights_IEGANnoD99.pth'
    G_net = IE_GPart()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5test(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        predict = G_net(batches_x)
        outname = output_path +str(step)+'.png'
        batch_x ,batch_y ,predict = torch.squeeze(batches_x,0).permute(1,2,0),torch.squeeze(batches_y,0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
        batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
        batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
        out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
        r = cal_psnr(predict,batch_y,255.)
        psnr_ += r
        if r < min_ :min_ = r
        if r > max_ :max_ = r
        cv2.imwrite(outname,out_all)
        count += 1
        print(r)
    print('num:',count,psnr_/count,min_,max_)

def test_AE_byh5():#原图410×640测试
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_AE'
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_AE/weights_AE99.pth'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    G_net = AE_()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5test(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        predict = G_net(batches_x)
        outname = output_path +'/'+str(step)+'.png'
        batch_x ,batch_y ,predict = torch.squeeze(batches_x,0).permute(1,2,0),torch.squeeze(batches_y,0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
        batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
        batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
        out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
        r = cal_psnr(predict,batch_y,255.)
        psnr_ += r
        if r < min_ :min_ = r
        if r > max_ :max_ = r
        cv2.imwrite(outname,out_all)
        count += 1
        print(r)
    print('num:',count,psnr_/count,min_,max_)

def test_zljplan1_byh5():#原图410×640测试
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_zljplan1'
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_zljplan1/99.pth'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    G_net = AE_zljplan1()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5test(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        predict = G_net(batches_x)
        outname = output_path +'/'+str(step)+'.png'
        batch_x ,batch_y ,predict = torch.squeeze(batches_x,0).permute(1,2,0),torch.squeeze(batches_y,0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
        batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
        batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
        out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
        r = cal_psnr(predict,batch_y,255.)
        psnr_ += r
        if r < min_ :min_ = r
        if r > max_ :max_ = r
        cv2.imwrite(outname,out_all)
        count += 1
        print(r)
    print('num:',count,psnr_/count,min_,max_)

def test_zljplan2_byh5():#原图410×640测试
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_zljplan2'
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_zljplan2/99.pth'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    G_net = AE_zljplan2()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5test(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        predict = G_net(batches_x)
        outname = output_path +'/'+str(step)+'.png'
        batch_x ,batch_y ,predict = torch.squeeze(batches_x,0).permute(1,2,0),torch.squeeze(batches_y,0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
        batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
        batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
        out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
        r = cal_psnr(predict,batch_y,255.)
        psnr_ += r
        if r < min_ :min_ = r
        if r > max_ :max_ = r
        cv2.imwrite(outname,out_all)
        count += 1
        print(r)
    print('num:',count,psnr_/count,min_,max_)

def test_zljplan3_byh5():#原图410×640测试
    img_h5_path='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5'
    label_h5_path ='/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_clean_test.h5'
    output_path = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/results_zljplan3'
    train_model = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/h5results/weights_zljplan3/99.pth'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    G_net = AE_zljplan3()
    G_net.load_state_dict(torch.load(train_model))
    G_net.eval()
    testset = infraredDatah5test(img_h5_path,label_h5_path)#内部实现了裁剪
    data_loader = DataLoader(testset,1)
    psnr_ , min_ ,max_ = 0,100,0
    count = 0
    for step,(batches_x,batches_y) in enumerate(data_loader):
        predict = G_net(batches_x)
        outname = output_path +'/'+str(step)+'.png'
        batch_x ,batch_y ,predict = torch.squeeze(batches_x,0).permute(1,2,0),torch.squeeze(batches_y,0).permute(1,2,0),torch.squeeze(predict,0).permute(1,2,0)
        batch_x ,batch_y ,predict = batch_x.detach().numpy(), batch_y.detach().numpy(),predict.detach().numpy()
        batch_x ,batch_y ,predict = batch_x*255 , batch_y*255 ,predict*255
        out_all = np.concatenate((batch_x,batch_y,predict),axis=1)
        r = cal_psnr(predict,batch_y,255.)
        psnr_ += r
        if r < min_ :min_ = r
        if r > max_ :max_ = r
        cv2.imwrite(outname,out_all)
        count += 1
        print(r)
    print('num:',count,psnr_/count,min_,max_)

if __name__ == "__main__":
    logdir = '/home/ipsg/code/ZLJ/Project/Infrared/torch_lua_IEGAN/IE-GAN_Pytorch_zlj/logs/AE_zljplan4/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    #  ---------------------sx h5----------------------------------------
    # train_model_byh5()
    # train_modelnoD_byh5(writer)
    # train_AE_byh5(writer)
    # train_zlj_plan1(writer)
    # train_zlj_plan2(writer)
    # train_zlj_plan3(writer)
    train_zlj_plan4(writer)

    # test_G_byimages()
    # test_GnoD_byimages()

    # test_G_byh5()
    # test_GnoD_byh5()
    # test_AE_byh5()
    # test_zljplan1_byh5()
    # test_zljplan2_byh5()
    # test_zljplan3_byh5()