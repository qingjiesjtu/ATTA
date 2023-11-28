from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import cv2
import argparse
from PIL import Image
import shutil
from io import BytesIO

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models_TS import *
from utils_TS.utils_TS import *
from utils_TS.datasets_TS import *
from utils_TS.utils_TS import rescale_boxes_TS
from utils_TS.datasets_TS import resize_TS

from models_OD import *
from utils_OD.utils_OD import *
from utils_OD.datasets_OD import *

from model_RL import SCNN
from utils_RL.utils import *
from utils_RL.prob2lines import getLane
from utils_RL.transforms import *
from  torchvision import utils as vutils

from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from ALL_sign_data.resnet import ResNet18
from tqdm import tqdm

from attention_RL import GradCAMTensor

import pandas as pd
import csv

import wandb

class SegmentationModelOutputWrapper(torch.nn.Module):
    # 改变模型的输出格式来匹配GradCAMTensor
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        self.exist_pred = None

    def forward(self, x):
        seg_pred, exist_pred = self.model(x)[:2]
        self.exist_pred = exist_pred.detach().cpu().numpy()
        return seg_pred

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

class ActivationsAndGradientsNotDetached:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class RemainTensor(EigenCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super().__init__(model, target_layers, use_cuda, reshape_transform)
        self.activations_and_grads = ActivationsAndGradientsNotDetached(
            self.model, target_layers, reshape_transform)
        
    def forward(self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False) -> torch.Tensor:
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if len(outputs)>1: outputs=outputs[0] # modified to adapt to yolov3
        if targets is None: # here to modify target class!!
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> torch.Tensor:
        activations_list = [a
                            for a in self.activations_and_grads.activations] # here required grad is False!!
        grads_list = [g
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(layer_activations)
            cam2 = []
            for ca in cam:
                cam2.append(torch.maximum(ca, torch.tensor(0)))
            cam = cam2
            # cam = torch.maximum(cam, 0)
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[0])

        return cam_per_target_layer

    def scale_cam_image(self,cam,target_size=None):
        result = []
        for img in cam:
            img = img - torch.min(img)
            img = img / (1e-7 + torch.max(img))
            if target_size is not None:
                img=img.reshape(1,1,img.shape[0],img.shape[1])
                img=F.interpolate(img,size=target_size,mode='bilinear')
                # img = img.resize(target_size)
            result.append(img)
        # result = result.type(torch.float32)
        return result

        
    def get_cam_image(self, activation_batch):
        # TBD: use pytorch batch svd implementation
        activation_batch[torch.isnan(activation_batch)] = 0
        projections = []
        for activations in activation_batch:
            reshaped_activations = (activations).reshape(
                activations.shape[0], -1).T
            # Centering before the SVD seems to be important here,
            # Otherwise the image returned is negative
            reshaped_activations = reshaped_activations - \
                reshaped_activations.mean(axis=0)
            U, S, VT = torch.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0, :]
            projection = projection.reshape(activations.shape[1:])
            projections.append(projection)
        return projections
    
    def aggregate_multi_layers(
            self,
            cam_per_target_layer: torch.tensor) -> torch.tensor:
        cam_per_target_layer = torch.cat(cam_per_target_layer, axis=1)
        cam_per_target_layer = torch.maximum(cam_per_target_layer, torch.tensor(0))
        result = torch.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

class MyOptimizer(Optimizer):
    def __init__(self, params, lr) -> None:
        # self.lr = lr
        super().__init__(params, {})
        self.param_groups[0]['lr']=lr
        
    def step(self, closure=False):
        for param_group in self.param_groups:
            params = param_group['params']
            lr = param_group['lr']
            # 从param_group中拿出参数
            for param in params:
                # 循环更新每一个参数的值
                param.data = np.clip(param.data - lr * param.grad,0,1)

def classify_draw_TS(dir_,img_show,img_class,detections,nums,step=1):
    fig, ax = plt.subplots()
    j = 0
    objects=[]
    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections): #  one object in a image
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        box_w = x2 - x1
        box_h = y2 - y1
        min_sign_size = 3
        size_show = int(100*img_show.size[0]/2048)
        # print("box 信息 x1,y1,x2,y2:\n",x1, y1,x2,y2)
        if box_w >= min_sign_size and box_h >= min_sign_size and conf >= 0.94:
            # print(conf,cls_conf)
            crop_sign_org = img_class.crop((x1-10, y1-10, x2+10, y2+10)).convert(mode="RGB")
            ##### to class  ###############
            test_transform = torchvision.transforms.Compose([ 
                torchvision.transforms.Resize((28, 28), interpolation=2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            crop_sign_input = test_transform(crop_sign_org).unsqueeze(0)
            with torch.no_grad():
                pred_class = model_class_TS(crop_sign_input.to(device))
            # 输出结果是所有类别的置信度，基本上是，正确的结果大于1，其余都是负数
            # print("分类模型输出结果： ",torch.max(pred_class, 1))
            sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]
            # print(cls_pred)
            # 更新值-------------------------------------------------------------------------------
            cls_pred = sign_type
            
            # print("cls_pred_type = ", classes_TS[int(cls_pred)])#," cls_conf = ",cls_conf)
            # #####
            # draw image 
            # #####
            if True and classes_TS[int(cls_pred)] != "zo":
                #  save predict results to a json file: my_train_results.json
                objects.append([int(cls_pred), x1, y1, x2, y2])

                color = "r"

                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)

                pad_sign_path_png = "ALL_sign_data/pad-all/" + classes_TS[int(cls_pred)] + ".png"
                pad_sign_path_jpg = "ALL_sign_data/pad-all/" + classes_TS[int(cls_pred)] + ".jpg"
                if  os.path.isfile(pad_sign_path_png):
                    pad_sign = Image.open(pad_sign_path_png)
                elif os.path.isfile(pad_sign_path_jpg):
                    pad_sign = Image.open(pad_sign_path_jpg)
                else:
                    pad_sign = Image.new("RGB", (size_show, size_show), (255, 255, 255))

                img_show.paste(crop_sign_org.resize((size_show, size_show)), (0, j * size_show) )
                img_show.paste(pad_sign.resize((size_show, size_show)), (size_show, j * size_show) )
                j += 1
                

    # Save generated image with detections
    ax.imshow(img_show)
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    img_upload = np.asarray(Image.open(buffer))

    plt.close()
    return objects,img_upload

def compute_loss(cam_now,cam_orient,option):
    '''
    option: 1->var;  2 -> + l2-norm(Differ);
    '''
    if option==1:
        # var
        loss = torch.var(cam_now,unbiased=False)
    elif option==2:
        # l2-norm(Differ) 
        loss = torch.norm(cam_now-cam_orient,p=2)
    elif option==3:
        loss = torch.norm(cam_now-cam_orient,p=1)
    else:
        loss = torch.var(cam_now,unbiased=False)
    return loss

def Iou_TS(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = max([xmin1, xmin2])
    yy1 = max([ymin1, ymin2])
    xx2 = min([xmax1, xmax2])
    yy2 = min([ymax1, ymax2])	
    # 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (max([0, xx2-xx1])) * (max([0, yy2-yy1])) #计算交集面积
    iou = inter_area / (area1+area2-inter_area+1e-6)    #计算交并比
    return iou

def Iou_OD(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = max([xmin1, xmin2])
    yy1 = max([ymin1, ymin2])
    xx2 = min([xmax1, xmax2])
    yy2 = min([ymax1, ymax2])	
    # 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (max([0, xx2-xx1])) * (max([0, yy2-yy1])) #计算交集面积
    iou = inter_area / (area1+area2-inter_area+1e-6)    #计算交并比

    return iou

def Miss_Generation_Misdetect_TS(obj_0,obj_i):
    # [int(cls_pred), x1, y1, x2, y2]
    len0=len(obj_0)
    leni=len(obj_i)
    cls0=[x[0] for x in obj_0]
    clsi=[x[0] for x in obj_i]
    x10 =[x[1] for x in obj_0]
    x1i =[x[1] for x in obj_i]
    y10 =[x[2] for x in obj_0]
    y1i =[x[2] for x in obj_i]
    x20 =[x[3] for x in obj_0]
    x2i =[x[3] for x in obj_i]
    y20 =[x[4] for x in obj_0]
    y2i =[x[4] for x in obj_i]
    miss=0
    generation=0
    misdetect=0
    if len0 >leni:
        miss=1
    elif len0 <leni:
        generation=1


    Ismisclass=[0]*(len0)
    IsMiss=[0]*(len0)
    Isgeneration=[0]*(len0)
    print("0-class: ",cls0,"i-class: ",clsi)
    for i in range(len0):
        i_mis=1
        i_misclass=0
        i_difpos=1
        for j in range(leni):
            iou=Iou_TS([x1i[j],y1i[j],x2i[j],y2i[j]],[x10[i],y10[i],x20[i],y20[i]])
            if iou>0.4: # 相同位置
                # input()
                i_difpos=0  # 相同位置应该认为是误判或者丢失，一定不是generation
                if clsi[j]!=cls0[i]:    #标签不同，相同位置,误判
                    i_misclass+=1
                else:   # 相同位置,相同标签,那么就没有miss
                    i_mis=0
        Ismisclass[i]=bool(i_misclass)
        IsMiss[i]=bool(i_mis)
        Isgeneration[i]=bool(i_difpos)
    
    miss=bool(miss+sum(IsMiss))
    generation=bool(generation+sum(Isgeneration))
    misdetect=bool(misdetect+sum(Ismisclass))
    return miss,misdetect,generation

def IsMisMisGen_OD(boxinfo_list0,n0,boxinfo_list,n):
    Ismisclass=[0]*(n0+1)
    IsMiss=[0]*(n0+1)
    Isgeneration=[0]*(n0+1)
    if n0>n:
        IsMiss[-1]=1
    if n0<n:
        Isgeneration[-1]=1
    for i in range(n0):
        i_mis=1
        i_misclass=0
        i_difpos=1
        [x1i, y1i, x2i, y2i, conf, cls_confi, cls_predi]=boxinfo_list0[i]
        for j in range(n):
            [x1j, y1j, x2j, y2j, conf, cls_confj, cls_predj]=boxinfo_list[j]
            iou=Iou_OD([x1i,y1i,x2i,y2i],[x1j,y1j,x2j,y2j])
            # print("iou = ",iou)
            if iou>0.4: # 相同位置
                i_difpos=0  # 相同位置应该认为是误判或者丢失，一定不是generation
                if cls_predi!=cls_predj:    #标签不同，相同位置
                    # print("位置相同,标签不同",i,j)
                    # input()
                    if cls_confi<cls_confj: #标签不同，位置相同，且置信度j更大->misclass i to j
                        i_misclass+=1
                else:   # 相同位置,相同标签,那么就没有miss
                    # print("位置相同,标签相同",i,j)
                    i_mis=0
            # print("位置不同",i,j)
        Ismisclass[i]=bool(i_misclass)
        IsMiss[i]=bool(i_mis)
        Isgeneration[i]=bool(i_difpos)
    
    return bool(sum(IsMiss)),bool(sum(Ismisclass)),bool(sum(Isgeneration))

def setposition_OD(boxinfo_list):
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    N=len(boxinfo_list)
    for i in range(N):
        [xx1, yy1, xx2, yy2, conf, cls_conf, cls_pred]=boxinfo_list[i]
        x1.append(xx1.cpu().numpy())
        y1.append(yy1.cpu().numpy())
        x2.append(xx2.cpu().numpy())
        y2.append(yy2.cpu().numpy())
    
    if N>0:
        idx=np.random.randint(low=0,high=N)
    else:
        idx=0
    
    if x2[idx]>365 and y1[idx]>230:
        pos_x1=np.random.randint(low=x1[idx]-opt.patch_size[0]-10,high=x1[idx]-opt.patch_size[0])
        pos_y1=np.random.randint(low=y1[idx],high=y2[idx]-20)
    else:
        pos_x1=np.random.randint(low=x1[idx]-opt.patch_size[0]-10,high=x2[idx]+10)
        pos_y1=np.random.randint(low=y2[idx],high=y2[idx]+10)
    
    iscox=0
    j=0
    for i in range(N):
        if x1[i]<=pos_x1+opt.patch_size[0] and x2[i]>=pos_x1 and y1[i]<=pos_y1+opt.patch_size[0] and y2[i]>=pos_y1:
            iscox=1

    while (iscox) or (pos_x1<0 or pos_x1+opt.patch_size[0]>416):
        j+=1
        print("Box crossed, try again.")
        if x1[idx]>280 and y1[idx]>230:
            pos_x1=np.random.randint(low=x1[idx]-opt.patch_size[0]-10,high=x1[idx]-opt.patch_size[0])
            pos_y1=np.random.randint(low=y1[idx],high=y2[idx]-20)
        else:
            pos_x1=np.random.randint(low=x1[idx]-opt.patch_size[0]-10,high=x2[idx]+10)
            pos_y1=np.random.randint(low=y2[idx],high=y2[idx]+10)
        
        if j%15==0 and N>0 :
            idx=(idx+1)%N

        iscox=0
        for i in range(N):
            if x1[i]<=pos_x1+80 and x2[i]>=pos_x1 and y1[i]<=pos_y1+40 and y2[i]>=pos_y1:
                iscox=1
    print("pos_y1: ",pos_y1,", pos_x1: ",pos_x1)
    return [pos_y1,pos_x1]

def setposition_TS(objs,img_size):
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    N=len(objs)
    for k in range(N):
        x1.append(int(objs[k][1]*opt.img_size_TS/img_size))
        y1.append(int(objs[k][2]*opt.img_size_TS/img_size))
        x2.append(int(objs[k][3]*opt.img_size_TS/img_size))
        y2.append(int(objs[k][4]*opt.img_size_TS/img_size))
    
    
    idx=np.random.randint(low=0,high=N)
    
    pos_x1=np.random.randint(low=x1[idx]-100,high=x2[idx]+100)
    pos_y1=np.random.randint(low=y2[idx],high=y2[idx]+30)

    iscox=0
    for i in range(N):
        if x1[i]<=pos_x1+opt.patch_size[0] and x2[i]>=pos_x1 and y1[i]<=pos_y1+opt.patch_size[0] and y2[i]>=pos_y1:
            iscox=1
    j=0
    isover=0
    if (pos_x1<0 or pos_x1+opt.patch_size[0]>opt.img_size_TS or pos_y1<0 or pos_y1+opt.patch_size[0]>opt.img_size_TS):
        isover=1
    while iscox or isover :
        j+=1
        print("Box crossed, try again.")
        pos_x1=np.random.randint(low=x1[idx]-90,high=x2[idx]+20)
        pos_y1=np.random.randint(low=y2[idx]+10,high=y2[idx]+30)
        if j%15==0:
            idx=(idx+1)%N
        
        iscox=0
        for i in range(N):
            if x1[i]<=pos_x1+opt.patch_size[0] and x2[i]>=pos_x1 and y1[i]<=pos_y1+opt.patch_size[0] and y2[i]>=pos_y1:
                iscox=1
        isover=0
        if (pos_x1<0 or pos_x1+opt.patch_size[0]>opt.img_size_TS or pos_y1<0 or pos_y1+opt.patch_size[0]>opt.img_size_TS):
            isover=1

    print("pos_y1: ",pos_y1,", pos_x1: ",pos_x1)
    return [pos_y1,pos_x1]

def load_img_OD(img_path):
    '''
    加载一张图片数据，并处理成model的输入格式;输入图片的路径
    '''
    img = np.array(Image.open(img_path))
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)    
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))  
    # 高度小于宽度时，按行左右填充；否则按列上下填充
    
    # 计算图像对应位置
    rect_xxyy=[0,0,416,416] # kuan_start,gao_start,kuan_end,gao_end
    
    if h<=w:
        rect_xxyy[1]=int(pad1/w * 416)
        rect_xxyy[3]=int((pad1+h)/w * 416)
    else:
        rect_xxyy[0]=int(pad1/h * 416)
        rect_xxyy[2]=int((pad2+w)/h * 416)

    img_hidden = np.pad(img, pad, 'constant', constant_values=127.5) / 255. # 归一化了
    
    print(img_hidden.shape,rect_xxyy)
    
    img_hidden=resize(img_hidden, (416, 416, 3), mode='reflect')    # 重新变化大小
    img_hidden = np.transpose(img_hidden, (2, 0, 1))
    img_hidden = torch.from_numpy(img_hidden).float()

    input_img = Variable(img_hidden.type(Tensor)).unsqueeze(0).to(device)
    return input_img,rect_xxyy

def parse_detections_OD(results,step):
    '''
    就是使用model进行检测并且对结果进行处理，返回识别框列表和标签信息列表
    '''
    # 存储每一个bbox并返回用于绘制图像
    bboxinfo_list=[]
    labelinfo_list=[]
    label_string_list=""
    if results[0] is not None:
        bbox_colors=colors
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in results[0]:
            print ('\t+ Label: %s, Conf: %.5f' % (classes_OD[int(cls_pred)], cls_conf.item()))
            label_string_list+="\t+ Label:"+ classes_OD[int(cls_pred)] +", Conf:"+str(cls_conf.item())+"<br>"
            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = bbox_colors[int(cls_pred)]
            # store bbox
            bboxinfo_list.append([x1, y1, x2, y2, conf, cls_conf, cls_pred])#,unpad_h,unpad_w,pad_x,pad_y])
            # store labelinfo
            labelinfo_list.append(color)
    # tb_writer.log({
    #     'Labellist':wandb.Html(label_string_list)
    # },step=step)
    return bboxinfo_list,labelinfo_list

def draw_detections_OD(bboxinfo_list, labelinfo_list,img):
    '''
    draw detected boxes
    '''
    plt.figure()
    fig, ax2 = plt.subplots(1)
    # 图片分辨率 = figsize*dpi 代码为416*416
    plt.rcParams['figure.figsize'] = (8.32, 8.32) 
    plt.rcParams['savefig.dpi'] = 50 
    
    ax2.imshow(img)
    
    for i in range(len(labelinfo_list)):
        [x1, y1, x2, y2, conf, cls_conf, cls_pred]=bboxinfo_list[i]
        color=labelinfo_list[i]
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        # Rescale coordinates to original dimensions
        box_h = int(((y2 - y1)))
        box_w = int(((x2 - x1)))
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                edgecolor=color,
                                facecolor='none')
        # print("bbox type:    ",type(bbox))
        ax2.add_patch(bbox)
        plt.text(x1, y1-10, s=classes_OD[int(cls_pred)]+' '+ str('%.4f'%cls_conf.item()), color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
    
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    new_img = np.asarray(Image.open(buffer))
    plt.close()
    return new_img

def renormalize_cam_with_bounding_OD(boxinfo_list, image_float_np, grayscale_cam, labelinfo_list=None):
    """
    Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. 
    当用来规范整图的cam时，需要注意boxinfo_list=[[x_start,y_start,x_end,y_end]]
    """
    if len(boxinfo_list) !=0:
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        # print(renormalized_cam.shape)
        for i in range(len(boxinfo_list)):
            x1=int(boxinfo_list[i][0])
            if x1<0:
                x1=0
            elif x1>415:
                x1=415
            y1=int(boxinfo_list[i][1])
            if y1<0:
                y1=0
            elif y1>415:
                y1=415
            x2=int(boxinfo_list[i][2])
            if x2<0:
                x2=0
            elif x2>415:
                x2=415
            y2=int(boxinfo_list[i][3])
            if y2<0:
                y2=0
            elif y2>415:
                y2=415

            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
            renormalized_cam = scale_cam_image(renormalized_cam)
            eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
            if labelinfo_list is not None:
                # 如果是需要画图的检测框内热力图绘制框，那么把框画上
                image_with_bounding_boxes = draw_detections_OD(boxinfo_list, labelinfo_list, eigencam_image_renormalized)
            else:
                # 如果只是对整体图片进行热力图切块，那么直接传回就好
                return eigencam_image_renormalized
    else:
        image_with_bounding_boxes=image_float_np
    return image_with_bounding_boxes

def visualization_OD(grayscale_cam_visual,img_visual,model,tensor,rect_xxyy,step):
    '''
    Detect the img and show with cam
    输入grayscale_cam_visual，img_visual，rect_xxyy,img_raw都是np矩阵
    '''
    
    # visualization 传入的rect_xxyy是一层的数组[],但是renormalize_cam_with_bounding需要两层的输入[[]],此时传回的是np矩阵
    cam_image=renormalize_cam_with_bounding_OD([rect_xxyy], img_visual, grayscale_cam_visual)
    
    with torch.no_grad():
        results = model(tensor)
        results = non_max_suppression_OD(results, 8, opt.conf_thres, opt.nms_thres)
        
    # 通过parse_detections从results中获取绘图信息
    bboxinfo_list,labelinfo_list= parse_detections_OD(results,step)
    
    #  传递到draw中进行绘制，并返回图像
    heatmap_detect=draw_detections_OD(bboxinfo_list,labelinfo_list, cam_image.copy())

    return bboxinfo_list,labelinfo_list,heatmap_detect

# -------------以下是RL使用的函数

def Draw(seg_pred,exist_pred,img,num,step,suffix):
    # 统计
    n_line=0
    
    # 绘图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    line_img = np.zeros_like(img)
    line_img_list=[]
    img_i=line_img.copy()
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 125, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    # line_img[coord_mask == (0 + 1)] = color[0]
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            # print("exit_pred 第",i,"个满足阈值")
            line_img[coord_mask == (i + 1)] = color[i]
            img_fori=img_i.copy()
            img_fori[coord_mask == (i + 1)] = color[i]
            line_img_list.append(img_fori)
            n_line+=1
    
    img_show = cv2.addWeighted(src1=line_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    
    return n_line,line_img_list,line_img,img_show

def DGF_detect(n_x,n_i,line_x_list,line_i_list):
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 125, 255], [0, 255, 255]], dtype='uint8')
    Disappear,Generate,False_detect=0,0,0
    print("n_x: ",n_x,"n_i: ",n_i)
    diff_list_minus=[]
    diff_list_posit=[]
    for i in range(n_i):
        diff=(line_i_list[i][:,:,1]/line_i_list[i][:,:,1].max()-line_x_list[i][:,:,1]/line_x_list[i][:,:,1].max())
        diff_list_minus.append(sum(sum(np.minimum(diff,0))))
        diff_list_posit.append(sum(sum(np.maximum(diff,0))))
        if sum(sum((line_i_list[i][:,:,1]/line_i_list[i][:,:,1].max())))==0:
            Disappear=1
    for i in range(len(diff_list_minus)):
        if diff_list_posit[i]>200:
            Generate=1
        elif diff_list_minus[i]<-200:
            Disappear=1
        elif diff_list_posit[i]>100:
            False_detect=1
    return Disappear,Generate,False_detect

def Get_CAM_ALL(x,gradCamTensor):
    with torch.no_grad(): 
        output = model_RL(x)
    
    output=output.cpu().numpy()
    coord_mask = np.argmax(output[0], axis=0) 
    K=output.shape[1]

    # cam_img_hidden=img
    grayscale_cam_all=torch.zeros([288,512]).to(device)
    for k in range(1,K):
        mask_float = np.float32(coord_mask == k)
        targets = [SemanticSegmentationTarget(k, mask_float)] 
        grayscale_cam = gradCamTensor(input_tensor=x,targets=targets)[0][0]
        grayscale_cam_all=torch.add(grayscale_cam_all,grayscale_cam)

    # # 统一cam并显示
    grayscale_cam_all=grayscale_cam_all/grayscale_cam_all.max()
    return grayscale_cam_all

def Run_ODTSRL():
    # cam设置
    target_layers_TS = [model_TS.module_list[-2]]
    cam_TS = RemainTensor(model_TS, target_layers_TS, use_cuda=True)
    target_layers_OD = [model_OD.module_list[-2]]
    cam_OD = RemainTensor(model_OD, target_layers_OD, use_cuda=True)
    target_layers = [model_RL.model.layer2[1]] #message_passing层不能加hook！
    gradCamTensor = GradCAMTensor(model=model_RL,target_layers=target_layers,use_cuda=torch.cuda.is_available())
   
    # 统计参量
    MissingASR_TS,MisdetectASR_TS,GenerationASR_TS,TotalASR_TS=0,0,0,0
    MissingASR_OD,MisdetectASR_OD,GenerationASR_OD,TotalASR_OD=0,0,0,0
    MissingASR_RL,MisdetectASR_RL,GenerationASR_RL,TotalASR_RL=0,0,0,0

    MisMisMis_ASR,MdtMdtMdt_ASR,GenGenGen_ASR,AttAASR,TotalASR=0,0,0,0,0
        
    
    # 同时对应的数据
    opt.image_folder_RL = crop_dirs_RL
    opt.image_folder_TS = crop_dirs_TS
    opt.image_folder_OD = crop_dirs_OD
    names_TS = os.listdir(opt.image_folder_TS)
    names_RL = os.listdir(opt.image_folder_RL)
    names_OD = os.listdir(opt.image_folder_OD)
    N=min(len(names_TS),len(names_OD),len(names_RL))
    print("N == ",N)
    for i in tqdm(range(N)):
        # 统计变量
        isMissing_TS, isMisdetection_TS, isGeneration_TS  = 0, 0, 0
        isMissing_OD, isMisdetection_OD, isGeneration_OD  = 0, 0, 0
        isMissing_RL, isMisdetection_RL, isGeneration_RL  = 0, 0, 0

        mismismis,mdtmdtmdt,gengengen,attaodtsrl=0,0,0,0

        loss_TS_list=[]
        loss_OD_list=[]
        loss_RL_list=[]
        loss_all_list=[]
        
        # 加载图像路径
        img_path_OD = os.path.join(opt.image_folder_OD, names_OD[i])
        img_path_TS = os.path.join(opt.image_folder_TS, names_TS[i])
        img_path_RL = os.path.join(opt.image_folder_RL, names_RL[i])

        # 加载OD图像
        input_img_OD,rect_xxyy_OD=load_img_OD(img_path_OD)
        # 加载TS图像
        input_img_TS = torchvision.transforms.ToTensor()(Image.open(img_path_TS).convert(mode="RGB"))
        input_img_TS, pads = pad_to_square(input_img_TS, 0) #显存增加100M
        show_img_size_TS=input_img_TS.shape[1]
        # 加载RL图像
        img_RL = cv2.imread(img_path_RL)
        img_RL = cv2.cvtColor(img_RL, cv2.COLOR_BGR2RGB)
        img_RL = transform_img({'img': img_RL})['img']
        x_RL = transform_to_net({'img': img_RL})['img']
        x_RL.unsqueeze_(0)
        x_RL=x_RL.to(device)


        # 进行无patch的检测与显示-----------------------------------------------------------------------------TS        
        img_show_TS = Variable(input_img_TS.type(Tensor)).to(device)
        input_img_TS = resize_TS(input_img_TS, opt.img_size_TS).unsqueeze(0) # 重新变化图片大小和格式变化
        input_img_TS = Variable(input_img_TS.type(Tensor)).to(device)
        
        # 生成原始的cam
        with torch.no_grad():
            grayscale_cam_TS = cam_TS(input_img_TS)[0]

        # visualization the img without patch
        grayscale_cam_TS_numpy =  grayscale_cam_TS.cpu().detach().numpy()    # to_numpy, easy visualization
        grayscale_cam_show_TS=cv2.resize(grayscale_cam_TS_numpy, dsize=(show_img_size_TS, show_img_size_TS), interpolation=cv2.INTER_LINEAR)
        
        # 用于展示的原始图片numpy格式
        img_show_TS = img_show_TS.cpu().detach().numpy()
        img_show_TS = np.transpose(img_show_TS,(1,2,0))   #img_visual是np数据类型
        
        # 将cam放在img上
        cam_image_TS = show_cam_on_image(img_show_TS, grayscale_cam_show_TS, use_rgb=True)
        
        with torch.no_grad():
            detections_TS = model_TS(input_img_TS)
            detections_TS = non_max_suppression_TS(detections_TS, opt.conf_thres, opt.nms_thres)[0]    
        print("---- TS step 0 ----")
        if  detections_TS is not None:
            detections_TS = rescale_boxes_TS(detections_TS, opt.img_size_TS, [show_img_size_TS,show_img_size_TS])
            # 画图在cam贴图上
            img_show = Image.fromarray(cam_image_TS)
            # 用于分类的图片
            img_class = input_img_TS.clone()
            img_class = img_class.squeeze(0).cpu().detach().numpy()
            img_class = np.transpose(img_class,(1,2,0))
            img_class = cv2.resize(img_class, dsize=(show_img_size_TS, show_img_size_TS), interpolation=cv2.INTER_LINEAR)
            img_class = img_class*255
            img_class = Image.fromarray(img_class.astype(np.uint8))    # 用于分类的图片
            obj_0,img_down_0 = classify_draw_TS(opt.image_folder_TS,img_show,img_class,detections_TS,i,1)
            print("obj_0: ",obj_0)
  
        else:
            print("这个图片没能检测出来目标")
            continue

        # 进行无patch的检测与显示-----------------------------------------------------------------------------OD
        grayscale_cam_OD = cam_OD(input_img_OD)[0]
        # grayscale_cam_OD = 1/(1+torch.exp(-15*(grayscale_cam_OD-0.4))) # soft masking

        # visualization the img without patch
        grayscale_cam_show_OD =  grayscale_cam_OD.cpu().detach().numpy()    # to_numpy, easy visualization
        img_show_OD = input_img_OD.squeeze(0).cpu().detach().numpy()
        img_show_OD = np.transpose(img_show_OD,(1,2,0))
        print("---- OD step 0 ----")
        boxinfo_list0,colorinfo_list0,heatmap_detect_OD=visualization_OD(grayscale_cam_show_OD,img_show_OD,model_OD,input_img_OD,rect_xxyy_OD,0)
        
                    
        if len(colorinfo_list0)==0:
            continue
    
        # 进行无patch的检测与显示--------------------------------------------RL     
        seg_pred, exist_pred = net(x_RL)[:2]
        seg_pred = seg_pred.detach().cpu().numpy()[0]
        exist_pred = exist_pred.detach().cpu().numpy()
        n_x,line_x_list,line_img_x,img_x_show=Draw(seg_pred,exist_pred,img_RL,i,0,"NoCam")
        
        line_img=cv2.cvtColor(line_img_x, cv2.COLOR_RGB2BGR)

        # ------------- 获得CAM--------------
        grayscale_cam_all=Get_CAM_ALL(x_RL,gradCamTensor)
        cam_image = show_cam_on_image(np.array(img_RL)/255, grayscale_cam_all.detach().cpu().numpy(), use_rgb=True)
        _,_,_,img_cam_x=Draw(seg_pred,exist_pred,cam_image,i,0,"WithCam")
    
    
        # 创建目标热力图----------------------------------------------------------------------------------------TSOD
        orient_cam_TS=grayscale_cam_TS.clone()
        orient_cam_TS[:,:]=torch.zeros(opt.img_size_TS,opt.img_size_TS)
        orient_cam_TS[opt.patch_pos_TS[0]:opt.patch_pos_TS[0]+opt.patch_size[1],opt.patch_pos_TS[1]:opt.patch_pos_TS[1]+opt.patch_size[0]]=torch.ones(opt.patch_size[1],opt.patch_size[0])

        orient_cam_OD=grayscale_cam_OD.clone()
        orient_cam_OD[:,:]=torch.zeros(opt.img_size_OD,opt.img_size_OD)
        orient_cam_OD[opt.patch_pos_OD[0]:opt.patch_pos_OD[0]+opt.patch_size[1],opt.patch_pos_OD[1]:opt.patch_pos_OD[1]+opt.patch_size[0]]=torch.ones(opt.patch_size[1],opt.patch_size[0])
        orient_cam_OD=orient_cam_OD[rect_xxyy_OD[1]:rect_xxyy_OD[3],rect_xxyy_OD[0]:rect_xxyy_OD[2]]
        
        cam_orient_tensor=grayscale_cam_all.clone()
        cam_orient_tensor[:,:]=torch.zeros(resize_shape[1],resize_shape[0])
        cam_orient_tensor[opt.patch_pos_RL[0]:opt.patch_pos_RL[0]+opt.patch_size[1],opt.patch_pos_RL[1]:opt.patch_pos_RL[1]+opt.patch_size[0]]=torch.ones(opt.patch_size[1],opt.patch_size[0])
 
        # ---------------------Patch创建---------
        # 新建patch贴图并初始化迭代参数
        patch = torch.rand(3,opt.patch_size[1],opt.patch_size[0])
        patch.requires_grad_(True)  # upgrade able
        lr = opt.lr
        optimizer = MyOptimizer([patch], lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=40)
        
        for step in range(1,opt.epochs+1):
            print(f"step {step} running!!")
            
            # --------------------------------------在TS下训练并求得loss_TS------------------------------------------
            inputi_TS=input_img_TS.clone()
            ismiss,ismisdetect,isgeneration=0,0,0
            # print("加载的inputi大小： ",inputi_TS.shape)
            # attach he patch to the img
            inputi_TS[:,:,opt.patch_pos_TS[0]:opt.patch_pos_TS[0]+opt.patch_size[1],opt.patch_pos_TS[1]:opt.patch_pos_TS[1]+opt.patch_size[0]]=patch
            grayscale_cami_TS = cam_TS(inputi_TS)[0] #似乎只在第一次调用时（子循环内）显存增加2G
            
            # visualization the img with patch
            grayscale_cam_TS_numpyi =  grayscale_cami_TS.cpu().detach().numpy()    # to_numpy, easy visualization
            grayscale_cam_showi_TS = cv2.resize(grayscale_cam_TS_numpyi, dsize=(show_img_size_TS, show_img_size_TS), interpolation=cv2.INTER_LINEAR)
            inputi_numpy_TS = inputi_TS.squeeze(0).cpu().detach().numpy()
            inputi_numpy_TS = np.transpose(inputi_numpy_TS,(1,2,0))
            inputi_numpy_TS = cv2.resize(inputi_numpy_TS, dsize=(show_img_size_TS, show_img_size_TS), interpolation=cv2.INTER_LINEAR)
            
            # 将cam放在img上
            cam_img_TS_i = show_cam_on_image(inputi_numpy_TS, grayscale_cam_showi_TS, use_rgb=True)
            
            # 进行检测
            with torch.no_grad():
                detections_TS_i = model_TS(inputi_TS)
                detections_TS_i = non_max_suppression_TS(detections_TS_i, opt.conf_thres, opt.nms_thres)[0]

            if  detections_TS_i is not None:
                detections_TS_i = rescale_boxes_TS(detections_TS_i, opt.img_size_TS, [show_img_size_TS,show_img_size_TS])

                # 画图在cam贴图上
                img_show_TS_i = Image.fromarray(cam_img_TS_i)
                inputi_numpy_TS = inputi_numpy_TS*255
                inputi_numpy_TS = Image.fromarray(inputi_numpy_TS.astype(np.uint8))    # 用于分类的图片
                    
                # print("cam图加载的格式： ",img_show.size)
                obj_i,img_down_i = classify_draw_TS(opt.image_folder_TS,img_show_TS_i,inputi_numpy_TS,detections_TS_i,i,step)
                print(obj_i)
                # print("检测返回列表",obj_i)
                ismiss_TS,ismisdetect_TS,isgeneration_TS=Miss_Generation_Misdetect_TS(obj_0,obj_i)
                print("ismiss: ",bool(ismiss_TS),"\n ismisdetect: ",bool(ismisdetect_TS),"\n isgeneration: ",bool(isgeneration_TS))
            else:
                ismiss_TS=1

            # 统计
            if isMissing_TS==0 and ismiss_TS==1:  # 之前没有检测到，现在检测到丢失
                isMissing_TS=1
                MissingASR_TS+=1

            if isGeneration_TS==0 and isgeneration_TS==1:
                isGeneration_TS=1
                GenerationASR_TS+=1

            if isMisdetection_TS==0 and ismisdetect_TS==1:
                isMisdetection_TS=1
                MisdetectASR_TS+=1

            # 计算Loss_TS
            loss_TS=compute_loss(grayscale_cami_TS,orient_cam_TS.detach(),2)

            
            # --------------------------------------在OD下训练并求得loss_OD------------------------------------------
            print(f"step {step} running!!")
            inputi_OD=input_img_OD.clone()

            # attach he patch to the img
            inputi_OD[:,:,opt.patch_pos_OD[0]:opt.patch_pos_OD[0]+opt.patch_size[1],opt.patch_pos_OD[1]:opt.patch_pos_OD[1]+opt.patch_size[0]]=patch # need to verify if only patch is updated!!  input's requires_grad is True
            grayscale_cam_OD = cam_OD(inputi_OD)[0] # adapt to nparray, need to prevent transformation from tensor to nparray in source code
            # grayscale_cam_OD = 1/(1+torch.exp(-20*(grayscale_cam_OD-0.4))) # soft masking

            # visualization
            grayscale_cam_visual =  grayscale_cam_OD.cpu().detach().numpy()
            img_visual = inputi_OD.squeeze(0).cpu().detach().numpy()
            img_visual = np.transpose(img_visual,(1,2,0))
            boxinfo_list,labelinfo_list,heatmap_detect=visualization_OD(grayscale_cam_visual,img_visual,model_OD,inputi_OD,rect_xxyy_OD,step)

            ismiss_OD,ismisclass_OD,isgeneration_OD=IsMisMisGen_OD(boxinfo_list0,len(colorinfo_list0),boxinfo_list,len(labelinfo_list))
            print("丢失：",bool(ismiss_OD),"\n"," 误判：",bool(ismisclass_OD),"\n"," 新产生：",bool(isgeneration_OD),"\n")

            
            if isMissing_OD==0 and ismiss_OD==1:  # 之前没有检测到，现在检测到丢失
                isMissing_OD=1
                MissingASR_OD+=1

            if isMisdetection_OD==0 and ismisclass_OD==1:
                isMisdetection_OD=1
                MisdetectASR_OD+=1

            if isGeneration_OD==0 and isgeneration_OD==1:
                isGeneration_OD=1
                GenerationASR_OD+=1

            cam_now_OD=grayscale_cam_OD[rect_xxyy_OD[1]:rect_xxyy_OD[3],rect_xxyy_OD[0]:rect_xxyy_OD[2]]
            loss_OD = compute_loss(cam_now_OD,orient_cam_OD.detach(),2)

            
            # --------------------------------------在RL下训练并求得loss_RL------------------------------------------
            imgi=img_RL.copy()
            # attach he patch to the img
            print(imgi.shape)
            
            xi = transform_to_net({'img':imgi})['img']
            xi.unsqueeze_(0)
            xi[:,:,opt.patch_pos_RL[0]:opt.patch_pos_RL[0]+opt.patch_size[1],opt.patch_pos_RL[1]:opt.patch_pos_RL[1]+opt.patch_size[0]]=patch*255
            xi=xi.to(device)

            imgi[opt.patch_pos_RL[0]:opt.patch_pos_RL[0]+opt.patch_size[1],opt.patch_pos_RL[1]:opt.patch_pos_RL[1]+opt.patch_size[0],:]=np.transpose(patch.detach().cpu().numpy(),(1,2,0))*255
            print(imgi.shape)
            # ---------------------------------------------- 检测
            seg_predi, exist_predi = net(xi)[:2]
            seg_predi = seg_predi.detach().cpu().numpy()[0]
            exist_predi = exist_predi.detach().cpu().numpy()
            n_i,line_xi_list,line_img_i,img_xi=Draw(seg_predi,exist_predi,imgi,i,step,"NoCam")

            disappeari,generatei,false_detecti=DGF_detect(n_x,n_i,line_x_list,line_xi_list)
            print("Disappear_RL: ",bool(disappeari),"\ngenerate: ",bool(generatei),"\nfalse_detect: ",bool(false_detecti))
            # ---------------------------------------------- 获取新CAM
            grayscale_cam_alli=Get_CAM_ALL(xi,gradCamTensor)    # 返回的是numpy值
            
            cam_imagei = show_cam_on_image(np.array(imgi)/255, grayscale_cam_alli.detach().cpu().numpy(), use_rgb=True)
            
            _,_,_,img_cam_xi=Draw(seg_predi,exist_predi,cam_imagei,i,step,"WithCam")

            if isMissing_RL==0 and disappeari==1:
                isMissing_RL=1
                MissingASR_RL+=1

            if isGeneration_RL==0 and generatei==1:
                isGeneration_RL=1
                GenerationASR_RL+=1

            if isMisdetection_RL==0 and false_detecti==1:
                isMisdetection_RL=1
                MisdetectASR_RL+=1

            
            if isMissing_RL==1 and isGeneration_RL==1 and isMisdetection_RL==1:
                break
            loss_RL=compute_loss(grayscale_cam_alli,cam_orient_tensor.detach(),3)


            if (ismiss_OD or ismisclass_OD or isgeneration_OD):
                OD_attack=1
            else:
                OD_attack=0
            if (ismiss_TS or ismisdetect_TS or isgeneration_TS):
                TS_attack=1
            else:
                TS_attack=0
            if (disappeari or generatei or false_detecti):
                RL_attack=1
            else:
                RL_attack=0
            
            if TS_attack==1 and OD_attack==1 and RL_attack==1 and attaodtsrl==0:
                attaodtsrl=1
                AttAASR+=1

            if ismiss_OD and disappeari and ismiss_TS and mismismis==0:
                mismismis=1
                MisMisMis_ASR+=1
            if ismisclass_OD and ismisdetect_TS and false_detecti and mdtmdtmdt==0:
                mdtmdtmdt=1
                MdtMdtMdt_ASR+=1
            if isgeneration_OD and isgeneration_TS and generatei and gengengen==0:
                gengengen=1
                GenGenGen_ASR+=1

            if isMissing_RL and isMisdetection_RL and isGeneration_RL and isMissing_OD and isMisdetection_OD and isGeneration_OD and isMissing_TS and isMisdetection_TS and isGeneration_TS:
                print("所有攻击均实现，当前step为： ",step)
                break
            
            # -------------------------------------------计算总loss并回传----------------------------
            print("Loss_OD is : ",loss_OD.item())
            print("Loss_TS is : ",loss_TS.item())
            print("Loss_RL is : ",loss_RL.item())
            
            
            loss=loss_TS+loss_OD*2/3+loss_RL/60
            loss_all_list.append(loss.item())
            print("loss_All:",loss.item())
            
            loss.backward()     # loss 1 in AoA
            
            optimizer.step()
            scheduler.step(loss)

            optimizer.zero_grad()
        
        if isMissing_OD or isMisdetection_OD or isGeneration_OD:
            TotalASR_OD+=1
        if isMissing_TS or isMisdetection_TS or isGeneration_TS:
            TotalASR_TS+=1
        if isMissing_RL  or isGeneration_RL  or isMisdetection_RL:
            TotalASR_RL+=1
        
        if (isMissing_OD and isMissing_TS and isMissing_RL)or (isMisdetection_OD and isMisdetection_TS and isMisdetection_RL ) or (isGeneration_OD and isGeneration_TS and isGeneration_RL):
            TotalASR+=1
     

    MissingASR_TS/=N
    MisdetectASR_TS/=N
    GenerationASR_TS/=N
    TotalASR_TS/=N
    MissingASR_OD/=N
    MisdetectASR_OD/=N
    GenerationASR_OD/=N
    TotalASR_OD/=N
    MissingASR_RL/=N
    MisdetectASR_RL/=N
    GenerationASR_RL/=N
    TotalASR_RL/=N
    
    MisMisMis_ASR/=N
    MdtMdtMdt_ASR/=N
    GenGenGen_ASR/=N
    AttAASR/=N
    TotalASR/=N

    
    file_note = open("MultiTaskTrain/ASR_odtsrl.txt", 'w')
    file_note.write("各种攻击ASR"+
          "\nMissingASR_TS: "+str(MissingASR_TS)+
          "\nMisdetectASR_TS: "+str(MisdetectASR_TS)+
          "\nGenerationASR_TS: "+str(GenerationASR_TS)+
          "\nTotalASR_TS: "+str(TotalASR_TS)+
          "\n"+
          "\nMissingASR_OD: "+str(MissingASR_OD)+
          "\nMisdetectASR_OD: "+str(MisdetectASR_OD)+
          "\nGenerationASR_OD: "+str(GenerationASR_OD)+
          "\nTotalASR_OD: "+str(TotalASR_OD)+
          "\n"+
          "\nMissingASR_RL: "+str(MissingASR_RL)+
          "\nMisdetectASR_RL: "+str(MisdetectASR_RL)+
          "\nGenerationASR_RL: "+str(GenerationASR_RL)+
          "\nTotalASR_RL: "+str(TotalASR_RL)+
          "\n"+
          "\nMisMisMis: "+str(MisMisMis_ASR)+
          "\nMdtMdtMdt_ASR: "+str(MdtMdtMdt_ASR)+
          "\nGenGenGen_ASR: "+str(GenGenGen_ASR)+
          "\nAttAASR: "+str(AttAASR)+
          "\nTotalASR: "+str(TotalASR)
          )
    file_note.close()

if __name__ == "__main__":

    # TS 模型参数
    sign_classes = 115
    classes_weights_path_TS = "MultiTaskTrain/ALL_sign_data/checkpoints/model_acc_97__class_115_epoch_10.pt"
    
    # OD 模型参数
    kitti_weights = 'MultiTaskTrain/weights_OD/yolov3-kitti.weights'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder_TS", type=str, default="MultiTaskTrain/demo/TSD", help="path to dataset")
    parser.add_argument("--model_def_TS", type=str, default="MultiTaskTrain/config_TS/ALL_DATA.cfg", help="path to model definition file")
    parser.add_argument("--weights_path_TS", type=str, default="MultiTaskTrain/checkpointsTS/yolov3_ckpt_33.pth", help="path to weights file")
    parser.add_argument("--class_path_TS", type=str, default="MultiTaskTrain/ALL_sign_data/ALL_data_in_2_train/names.txt", help="path to class label file")

    parser.add_argument('--image_folder_OD', type=str, default='MultiTaskTrain/demo/OD', help='path to dataset')
    parser.add_argument('--config_path_OD', type=str, default='MultiTaskTrain/config_OD/yolov3-kitti.cfg', help='path to model config file')
    parser.add_argument('--weights_path_OD', type=str, default=kitti_weights, help='path to weights file')
    parser.add_argument('--class_path_OD', type=str, default='MultiTaskTrain/data_OD/kitti.names', help='path to class label file')
    
    parser.add_argument("--image_path_RL", type=str, default="MultiTaskTrain/demo/RL/96.jpg", help="Path to demo img")
    parser.add_argument("--image_folder_RL", type=str, default="MultiTaskTrain/demo/RL", help="Path to demo img")
    parser.add_argument("--weight_path_RL", type=str, default="MultiTaskTrain/weight_RL/exp0/exp0_best.pth",help="Path to model weights")
    
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    
    parser.add_argument("--img_size_TS", type=int, default=512, help="size of each image dimension")
    parser.add_argument('--img_size_OD', type=int, default=416, help='size of each image dimension')
    
    # 其他训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=100, help='learning rate')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[40,40], help='[min_train, max-train, test]')# [y,x]
    parser.add_argument('--patch_pos_TS', nargs='+', type=int, default=[270,100], help='[min_train, max-train, test]') 
    parser.add_argument('--patch_pos_OD', nargs='+', type=int, default=[270,100], help='[min_train, max-train, test]') 
    parser.add_argument('--patch_pos_RL', nargs='+', type=int, default=[200,215], help='[min_train, max-train, test]') 

    opt = parser.parse_args()

    print(opt)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Set up TS model
    model_TS = Darknet_TS(opt.model_def_TS, img_size=opt.img_size_TS).to(device)

    model_TS.load_state_dict(torch.load(opt.weights_path_TS))

    classes_TS = load_classes_TS(opt.class_path_TS)  # Extracts class labels from file

    # to  class
    model_class_TS = ResNet18(sign_classes)
    model_class_TS.load_state_dict(torch.load(classes_weights_path_TS))
    model_class_TS.to(device)
    model_class_TS.eval()

    # pics dir
    crop_dirs_TS = "MultiTaskTrain/demo/TSD"
    crop_dirs_OD = "MultiTaskTrain/demo/OD"
    crop_dirs_RL = "MultiTaskTrain/demo/RL"
    
    
    # set up OD model
    cmap = plt.get_cmap('tab20b')   # 设置框的颜色
    colors = [cmap(i) for i in np.linspace(0, 1, 100)]

    model_OD = Darknet_OD(opt.config_path_OD, img_size=opt.img_size_OD)
    model_OD.load_weights(opt.weights_path_OD)
    print('model path: ' +opt.weights_path_OD)
    model_OD.cuda()

    model_OD.eval() # Set in evaluation mode
    
    classes_OD = load_classes_OD(opt.class_path_OD) # Extracts class labels from file

    # 一些模参数
    resize_shape=(512, 288)
    net = SCNN(input_size=resize_shape, pretrained=False)
    mean=(0.3598, 0.3653, 0.3662) # CULane mean, std
    std=(0.2573, 0.2663, 0.2756)
    transform_img = Resize(resize_shape)
    transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))
    
    # 加载model
    save_dict = torch.load(opt.weight_path_RL, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.to(device)
    net.eval()

    model_RL = SegmentationModelOutputWrapper(net)

    Run_ODTSRL()  
