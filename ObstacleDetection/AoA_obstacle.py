from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
from models import *
from utils.utils import *
from utils.datasets import *
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import argparse
import cv2
from PIL import Image
from io import BytesIO
import datetime
import shutil

import torch
from torch.optim import Optimizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Callable, List, Tuple
import wandb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

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

def parse_detections(results,step):
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
            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            label_string_list+="\t+ Label:"+ classes[int(cls_pred)] +", Conf:"+str(cls_conf.item())+"<br>"
            color = bbox_colors[int(cls_pred)]
            # store bbox
            bboxinfo_list.append([x1, y1, x2, y2, conf, cls_conf, cls_pred])#,unpad_h,unpad_w,pad_x,pad_y])
            # store labelinfo
            labelinfo_list.append(color)
    return bboxinfo_list,labelinfo_list

def draw_detections(bboxinfo_list, labelinfo_list,img):
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
        ax2.add_patch(bbox)
        plt.text(x1, y1-10, s=classes[int(cls_pred)]+' '+ str('%.4f'%cls_conf.item()), color='white', verticalalignment='top',
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

def renormalize_cam_with_bounding(boxinfo_list, image_float_np, grayscale_cam, labelinfo_list=None):
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
                image_with_bounding_boxes = draw_detections(boxinfo_list, labelinfo_list, eigencam_image_renormalized)
            else:
                # 如果只是对整体图片进行热力图切块，那么直接传回就好
                return eigencam_image_renormalized
    else:
        image_with_bounding_boxes=image_float_np
    return image_with_bounding_boxes

def visualization(grayscale_cam_visual,img_visual,model,tensor,rect_xxyy,step):
    '''
    Detect the img and show with cam
    输入grayscale_cam_visual，img_visual，rect_xxyy,img_raw都是np矩阵
    '''
    # visualization 传入的rect_xxyy是一层的数组[],但是renormalize_cam_with_bounding需要两层的输入[[]],此时传回的是np矩阵
    cam_image=renormalize_cam_with_bounding([rect_xxyy], img_visual, grayscale_cam_visual)
    
    with torch.no_grad():
        results = model(tensor)
        results = non_max_suppression(results, 8, opt.conf_thres, opt.nms_thres)
        
    # 通过parse_detections从results中获取绘图信息
    bboxinfo_list,labelinfo_list= parse_detections(results,step)
    
    #  传递到draw中进行绘制，并返回图像
    heatmap_detect=draw_detections(bboxinfo_list,labelinfo_list, cam_image.copy())

    return bboxinfo_list,labelinfo_list,heatmap_detect

def load_img(img_path):
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

def setposition(boxinfo_list,ori_y):
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
        pos_x1=np.random.randint(low=x1[idx]-50,high=x1[idx]-40)
        pos_y1=np.random.randint(low=y1[idx],high=y2[idx]-20)
    else:
        pos_x1=np.random.randint(low=x1[idx]-10,high=x2[idx]+10)
        pos_y1=np.random.randint(low=y2[idx],high=y2[idx]+10)
    
    iscox=0
    j=0
    for i in range(N):
        if x1[i]<=pos_x1+40 and x2[i]>=pos_x1 and y1[i]<=pos_y1+40 and y2[i]>=pos_y1:
            iscox=1

    while (iscox) or (pos_x1<0 or pos_x1+40>416):
        j+=1
        print("Box crossed, try again.")
        if x1[idx]>280 and y1[idx]>230:
            pos_x1=np.random.randint(low=x1[idx]-50,high=x1[idx]-40)
            pos_y1=np.random.randint(low=y1[idx],high=y2[idx]-20)
        else:
            pos_x1=np.random.randint(low=x1[idx]-10,high=x2[idx]+10)
            pos_y1=np.random.randint(low=y2[idx],high=y2[idx]+10)
        
        if j%15==0 and N>0 :
            idx=(idx+1)%N
            
        iscox=0
        for i in range(N):
            if x1[i]<=pos_x1+80 and x2[i]>=pos_x1 and y1[i]<=pos_y1+40 and y2[i]>=pos_y1:
                iscox=1
    print("pos_y1: ",pos_y1,", pos_x1: ",pos_x1)
    return [pos_y1,pos_x1]

def Iou(box1, box2, wh=False):
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

def IsMisMisGen(boxinfo_list0,n0,boxinfo_list,n):
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
            iou=Iou([x1i,y1i,x2i,y2i],[x1j,y1j,x2j,y2j])
            # print("iou = ",iou)
            if iou>0.3: # 相同位置
                i_difpos=0  # 相同位置应该认为是误判或者丢失，一定不是generation
                if cls_predi!=cls_predj:    #标签不同，相同位置
                    if cls_confi<cls_confj: #标签不同，位置相同，且置信度j更大->misclass i to j
                        i_misclass+=1
                else:   # 相同位置,相同标签,那么就没有miss
                    i_mis=0
        Ismisclass[i]=bool(i_misclass)
        IsMiss[i]=bool(i_mis)
        Isgeneration[i]=bool(i_difpos)
    print("丢失：",bool(sum(IsMiss)),"\n"," 误判：",bool(sum(Ismisclass)),"\n"," 新产生：",bool(sum(Isgeneration)),"\n")
    return bool(sum(IsMiss)),bool(sum(Ismisclass)),bool(sum(Isgeneration))

def compute_loss(cam_now,cam_want,option):
    '''
    option: 1->var;   2-> l2-norm(Differ); 
    '''
    if option==1:
        # var
        loss = torch.var(cam_now,unbiased=False)
    elif option==2:
        # l2-norm(Differ) 
        loss = torch.norm(cam_now-cam_want,p=2)
    else:
        loss = torch.var(cam_now,unbiased=False)
    return loss

def Run():
    # Attention Map的目标层
    target_layers = [model.module_list[-2]]
    cam = RemainTensor(model, target_layers, use_cuda=True)

    # 统计
    MissingASR, MisdetectionASR, GenerationASR, TotalASR=0,0,0,0
    N=len(dataloader)
    L=N

    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # 统计
        isMissing, isMisdetection, isGeneration  = 0, 0, 0
        
        # 加载detect的图片,并完成对大小的处理，返回variable类型;返回第二个值为原照片在416大小变换图的位置信息
        img_path = ''.join(img_paths)
        print(img_path)
        input_img,rect_xxyy=load_img(img_path)

        # 克隆一份img
        img_ori=input_img.clone()
        grayscale_cam = cam(img_ori)[0]
        grayscale_cam = 1/(1+torch.exp(-15*(grayscale_cam-0.4))) # soft masking
        
        # visualization the img without patch
        grayscale_cam_visual =  grayscale_cam.cpu().detach().numpy()    # to_numpy, easy visualization  
        
        img_visual = img_ori.squeeze(0).cpu().detach().numpy()
        img_visual = np.transpose(img_visual,(1,2,0))
        print("step 0 running!!")
        boxinfo_list0,colorinfo_list0,heatmap_detect=visualization(grayscale_cam_visual,img_visual,model,img_ori,rect_xxyy,0)
        im = Image.fromarray(heatmap_detect)
        im.save(opt.output_path+'/D'+str(batch_i)+'_NoPatch.png')

        # generate a patch in random way, a 3xHxW size tensor each channel is a RGB
        patch = torch.rand(3,opt.patch_size[1],opt.patch_size[0])
        patch.requires_grad_(True)  # upgrade able
        lr = opt.lr
        optimizer = MyOptimizer([patch], lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=40)
        if len(colorinfo_list0)==0:
            continue
        patch_pos=setposition(boxinfo_list0,rect_xxyy[-1])
        
        # 产生注意力全在patch上的cam
        patch_cam_tensor=grayscale_cam.clone()
        patch_cam_tensor[:,:]=torch.zeros(416,416)
        patch_cam_tensor[patch_pos[0]:patch_pos[0]+opt.patch_size[1],patch_pos[1]:patch_pos[1]+opt.patch_size[0]]=torch.ones(opt.patch_size[1],opt.patch_size[0])
        patch_cam_tensor=patch_cam_tensor[rect_xxyy[1]:rect_xxyy[3],rect_xxyy[0]:rect_xxyy[2]]
        
        
        # new patch attack
        for step in range(1,opt.epochs+1):
            print(f"step {step} running!!")
            inputi=input_img.clone() #这个需要放到循环内部来，否则loss会backward两次(不知道为什么。。。）!!

            # attach he patch to the img
            inputi[:,:,patch_pos[0]:patch_pos[0]+opt.patch_size[1],patch_pos[1]:patch_pos[1]+opt.patch_size[0]]=patch # need to verify if only patch is updated!!  input's requires_grad is True
            grayscale_cam = cam(inputi)[0] # adapt to nparray, need to prevent transformation from tensor to nparray in source code
            grayscale_cam = 1/(1+torch.exp(-20*(grayscale_cam-0.4))) # soft masking

            # visualization
            grayscale_cam_visual =  grayscale_cam.cpu().detach().numpy()
            img_visual = inputi.squeeze(0).cpu().detach().numpy()
            img_visual = np.transpose(img_visual,(1,2,0))
            boxinfo_list,labelinfo_list,heatmap_detect=visualization(grayscale_cam_visual,img_visual,model,inputi,rect_xxyy,step)
            
            if step==1:
                im = Image.fromarray(heatmap_detect)
                im.save(opt.output_path+'/D'+str(batch_i)+'_FirstPatch.png')
                
            ismiss,ismisclass,isgeneration=IsMisMisGen(boxinfo_list0,len(colorinfo_list0),boxinfo_list,len(labelinfo_list))

            if isMissing==0 and ismiss==1:  # 之前没有检测到，现在检测到丢失
                isMissing=1
                MissingASR+=1
                im = Image.fromarray(heatmap_detect)
                im.save(opt.output_path+'/D'+str(batch_i)+'_Missing.png')
            if isMisdetection==0 and ismisclass==1:
                isMisdetection=1
                MisdetectionASR+=1
                im = Image.fromarray(heatmap_detect)
                im.save(opt.output_path+'/D'+str(batch_i)+'_Misdetect.png')
            if isGeneration==0 and isgeneration==1:
                isGeneration=1
                GenerationASR+=1
                im = Image.fromarray(heatmap_detect)
                im.save(opt.output_path+'/D'+str(batch_i)+'_Generation.png')
            
            if isMissing and isMisdetection and isGeneration:
                print("所有攻击均实现，当前step为： ",step)
                break
                
            cam_now=grayscale_cam[rect_xxyy[1]:rect_xxyy[3],rect_xxyy[0]:rect_xxyy[2]]
            loss = compute_loss(cam_now,patch_cam_tensor.detach(),2)
            
            loss.backward(retain_graph=True)     # loss 1 in AoA
            
            optimizer.step()
            scheduler.step(loss)
            # tb_writer.log({'patch grad l1':abs(patch.grad).sum(),
            #             'patch grad max':abs(patch.grad).max()},step=step)
            optimizer.zero_grad()
        
        if isMissing or isMisdetection or isGeneration:
            TotalASR+=1

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))
        if batch_i>L:
            break

    MissingASR/=L
    MisdetectionASR/=L
    GenerationASR/=L
    TotalASR/=L
    print("各种攻击ASR",
          "\nMissingASR: ",MissingASR,
          "\nMisdetectionASR: ",MisdetectionASR,
          "\nGenerationASR: ",GenerationASR,
          "\nTotalASR: ",TotalASR)
    file_note = open("ObstacleDetection/ASR.txt", 'w')
    file_note.write("各种攻击ASR"+
          "\nMissingASR: "+str(MissingASR)+
          "\nMisdetectionASR: "+str(MisdetectionASR)+
          "\nGenerationASR: "+str(GenerationASR)+
          "\nTotalASR: "+str(TotalASR))
    file_note.close()

if __name__ == '__main__':
    kitti_weights = 'ObstacleDetection/weights/yolov3-kitti.weights'
    # 参数列表
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='ObstacleDetection/data/imgtry/', help='path to dataset')
    parser.add_argument('--output_path', type=str, default='ObstacleDetection/outputOD', help='path to model config file')
    parser.add_argument('--config_path', type=str, default='ObstacleDetection/config/yolov3-kitti.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default=kitti_weights, help='path to weights file')
    parser.add_argument('--class_path', type=str, default='ObstacleDetection/data/kitti.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')    # 这里的img size意思是 416x416
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')  
    # 其他训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1000, help='learning rate')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[20,20], help='[min_train, max-train, test]')# [y,x]
    opt = parser.parse_args()
    print('Config:',opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available() and opt.use_cuda
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    transform = transforms.ToTensor()

    if os.path.exists(opt.output_path):
        print('\n \n Exist a folder alredy. Delete and remake.\n \n')
        shutil.rmtree(opt.output_path)#删除再建立
        os.makedirs(opt.output_path)
    else:
        os.makedirs(opt.output_path)
    
    # 绘制检测图
    cmap = plt.get_cmap('tab20b')   # 设置框的颜色
    colors = [cmap(i) for i in np.linspace(0, 1, 100)]
    # 加载模型
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    print('model path: ' +opt.weights_path)
    model.cuda()
    model.eval() # Set in evaluation mode
    
    # 加载数据
    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    print('data size : %d' % len(dataloader) )
    
    # 加载标签
    classes = load_classes(opt.class_path) # Extracts class labels from file
    print(classes)
    Run()
    
