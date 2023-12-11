import itertools
import numpy as np
import torch
from torchsummary import summary
import random
import time
from torchvision import datasets, transforms
import torch.utils.data as data
import torchvision.models as models
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datetime
from util import *
from PIL import Image
import shutil
import torch.nn.parallel
import torch.optim
import random
import numbers
import torch.nn.functional as F
import resnet_pose_attention_v2 as resnet
import option


class Config_orth:
    def __init__(self):
        self.data = "./dataset/"  # Adjust path as necessary
        self.range = 5
        self.workers = 4
        self.batch_size = 8
		
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")		

def to_one_hot(targets, num_classes):
    # Create a one-hot tensor on the correct device
    one_hot = torch.eye(num_classes, device=targets.device)
    # Index with targets to create the one-hot encoded target
    return one_hot[targets]
	
def main():

    # Initialize the config
    args = Config_orth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    l=args.range

    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model_cla = resnet.resnet50()
    model_cla = torch.nn.DataParallel(model_cla).cuda()
    model_cla.to(device)
    checkpoint = torch.load('/kaggle/input/la-qvit-orfe0-1/checkpoint_cnn/[11-18]-[10-35]-model_best.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    for k, v in pre_trained_dict.items():
        print(k, v.shape)
    model_cla.load_state_dict(pre_trained_dict)

    model_cla.eval()
    feature_1 = []
    feature_2 = []
    feature_3 = []
    label = []
    num_classes= 7
    
    top1 = AverageMeter('Accuracy', ':6.3f')

    with torch.no_grad():
        data_name = [item[0] for item in data_facial]
        for i, (images, target, fn) in enumerate(val_loader):

            # search
            facial_indx = []
            for j in range(len(fn)):
                facial_indx.append(data_name.index(fn[j]))
            facial=data_facial[facial_indx,1]
            facial = np.stack(facial, axis=0)
            images,rect,rect_local= pre_pro(images,facial,0.8,0.5,l,args.workers)

            images = images.cuda()
            target = target.cuda()
            
            #  if set in val_loader
            target = to_one_hot(target, num_classes)

            model_cla.module.set_rect(rect)
            model_cla.module.set_rect_local(rect_local)


            # compute output
            x_gf_1,x_gf_2,x_gf_3,x_gf_fc1,x_gf_fc2,x_gf_fc3, out_gf = model_cla(images)
            

            # measure accuracy and record loss
            acc1, _ = accuracy(out_gf, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))



            x_gf_1 = x_gf_1.permute(0, 2, 3, 1)
            x_gf_2 = x_gf_2.permute(0, 2, 3, 1)
            x_gf_3 = x_gf_3.permute(0, 2, 3, 1)


            if i == 0:

                feature_gf_1 = x_gf_1.cpu().numpy()
                feature_gf_2 = x_gf_2.cpu().numpy()
                feature_gf_3 = x_gf_3.cpu().numpy()
                
                _, turget = torch.max(target, 1)
                label = target.cpu().numpy()
                
            else:
                feature_gf_1 = np.concatenate((feature_gf_1, x_gf_1.cpu().numpy()),axis=0)
                feature_gf_2 = np.concatenate((feature_gf_2, x_gf_2.cpu().numpy()),axis=0)
                feature_gf_3 = np.concatenate((feature_gf_3, x_gf_3.cpu().numpy()),axis=0)
                _, turget = torch.max(target, 1)
                label = np.concatenate((label, target.cpu().numpy()),axis=0)


    print(' *** Accuracy {top1.avg:.3f}  *** '.format(top1=top1))
    # train
#     np.save("/kaggle/working/Orthognal_npy/train_gf_1_RAFDB2_v2.npy",feature_gf_1)
#     np.save("/kaggle/working/Orthognal_npy/train_gf_2_RAFDB2_v2.npy",feature_gf_2)
#     np.save("/kaggle/working/Orthognal_npy/train_gf_3_RAFDB2_v2.npy",feature_gf_3)    
#     np.save("/kaggle/working/Orthognal_npy/train_label_RAFDB2_v2.npy",label)
       
    # # test
    np.save("/kaggle/working/Orthognal_npy/test_gf_1_RAFDB2_v2.npy",feature_gf_1)
    np.save("/kaggle/working/Orthognal_npy/test_gf_2_RAFDB2_v2.npy",feature_gf_2)
    np.save("/kaggle/working/Orthognal_npy/test_gf_3_RAFDB2_v2.npy",feature_gf_3)
    np.save("/kaggle/working/Orthognal_npy/test_label_RAFDB2_v2.npy",label)	
	
if __name__ == '__main__':
    main()	