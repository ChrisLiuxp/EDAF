#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import efficientdet_dataset_collate, EfficientdetDataset
from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import Generator, FocalLoss
from nets.fcos_training import FCOSLoss
from tqdm import tqdm
from utils.vocdataset import VOCDataPrefetcher

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def fit_one_epoch(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    cls_losses, reg_losses, center_ness_losses, losses, val_loss = [], [], [], [], []

    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        prefetcher = VOCDataPrefetcher(gen)
        images, annotations = prefetcher.next()
        while images is not None:
            images, annotations = images.cuda().float(), annotations.cuda()
            cls_heads, reg_heads, center_heads, batch_positions = net(images)
            cls_loss, reg_loss, center_ness_loss = fcos_loss(
                cls_heads, reg_heads, center_heads, batch_positions, annotations)
            loss = cls_loss + reg_loss + center_ness_loss
            if cls_loss == 0.0 or reg_loss == 0.0:
                optimizer.zero_grad()
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            cls_losses.append(cls_loss.item())
            reg_losses.append(reg_loss.item())
            center_ness_losses.append(center_ness_loss.item())
            losses.append(loss.item())
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Conf Loss'         : cls_loss.item(),
                                'Regression Loss'   : reg_loss.item(),
                                'Center-ness Loss'  : center_ness_loss.item(),
                                'lr'                : get_lr(optimizer),
                                'step/s'            : waste_time})
            pbar.update(1)

            start_time = time.time()

            images, annotations = prefetcher.next()


    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        prefetcher_val = VOCDataPrefetcher(genval)
        images_val, annotations_val  = prefetcher_val.next()
        while images_val is not None:
            images_val, annotations_val = images_val.cuda().float(), annotations_val.cuda()
            with torch.no_grad():
                cls_heads, reg_heads, center_heads, batch_positions = net(images_val)
                cls_loss, reg_loss, center_ness_loss = fcos_loss(
                    cls_heads, reg_heads, center_heads, batch_positions, annotations_val)
                loss = cls_loss + reg_loss + center_ness_loss

                optimizer.zero_grad()

                val_loss.append(loss.item())

            pbar.set_postfix(**{'Total Loss'         : loss.item()})
            pbar.update(1)

            images_val, annotations_val  = prefetcher_val.next()

    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (losses/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),losses/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)
#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    #-------------------------------------------#
    phi = 0
    Cuda = False
    annotation_path = '2007_train.txt'
    classes_path = 'model_data/voc_classes.txt'   
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    input_shape = (input_sizes[phi], input_sizes[phi])

    # 创建模型
    model = EfficientDetBackbone(num_classes,phi)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    # model_path = "model_data/efficientdet-d0.pth"
    # # 加快模型训练的效率
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # efficient_loss = FocalLoss()
    fcos_loss = FCOSLoss()

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-3
        Batch_size = 1
        Init_Epoch = 0
        Freeze_Epoch = 50
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        if Use_Data_Loader:
            train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
            val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate()

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-4
        Batch_size = 1
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        if Use_Data_Loader:
            train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
            val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate()

                        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
