import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
# divide the image into SxS grid cells
# each grid cell predicts :
# B bounding boxes---x , y , w , h , confidence
# C conditional class probabilities---Pr(Class_i|Object)

# each bounding box consists of 5 predictions :
#  x , y , w , h , confidence
    # (x,y) : the center of the box relative to the bounds of the grid cell
    # w , h : width and height predicted relative to the whole image
    # confidence : Pr(Object)*IOU_{pred}^{truth}

# class-specific confidence scores for each box:
# Pr(Class_i|Object)*Pr(Object)*IOU_{pred}^{truth}=Pr(Class_i)*IOU_{pred}^{truth}
# indicate the probability of that class apearing in the box
# and how well the predicted box fits the object


# final prediction is an SxSx(B*5+C) tensor
# while evaluate the model on the PASCAL VOC detection dateset
S=7
B=2
C=20

lambda_coord=5.0
lambda_noobj=0.5
# the initial convolutional layers of the network extract features from the image
# the fully connected layers predict the output probabilities and coordinates

use_cuda=torch.cuda.is_available()
FloatTensor=torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor=torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor=FloatTensor

class BasicConv2d(nn.Module):
    def __init__(self,convs_param,if_pool,if_bn=False):
        super(BasicConv2d,self).__init__()
        convs=collections.OrderedDict()
        for idx,(in_c,out_c,kernel_size,stride,padding) in enumerate(convs_param):
            convs['conv%s'%(idx+1)]=nn.Conv2d(in_c,out_c,kernel_size,stride,padding)
            if if_bn:
                convs['batchnorm%s'%(idx+1)]=nn.BatchNorm2d(out_c)
            convs['leakyrelu%s'%(idx+1)]=nn.LeakyReLU(negative_slope=0.1)
        self.convs=nn.Sequential(convs)
        self.if_pool=if_pool
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        x=self.convs(x)
        if self.if_pool:
            x=self.maxpool(x)
        return x


class yolo(nn.Module):
    def __init__(self,S,B,C):
        super(yolo,self).__init__()
        self.output_dim=S*S*(B*5+C)
        self.layers1=BasicConv2d([(3,64,7,2,3)],True)
        self.layers2=BasicConv2d([(64,192,3,1,1)],True)
        self.layers3=BasicConv2d([(192,128,1,1,0),(128,256,3,1,1),
                                 (256,256,1,1,0),(256,512,3,1,1)],True)
        self.layers4=BasicConv2d([(512,256,1,1,0),(256,512,3,1,1),
                                 (512, 256, 1, 1,0), (256, 512, 3, 1,1),
                                 (512, 256, 1, 1,0), (256, 512, 3, 1,1),
                                 (512, 256, 1, 1,0), (256, 512, 3, 1,1),
                                 (512,512,1,1,0),(512,1024,3,1,1)
                                 ],True)
        self.layers5=BasicConv2d([(1024,512,1,1,0),(512,1024,3,1,1),
                                 (1024, 512, 1, 1,0), (512, 1024, 3, 1,1),
                                 (1024,1024,3,1,1),(1024,1024,3,2,1)
                                 ],False)
        self.layers6=BasicConv2d([(1024,1024,3,1,1),(1024,1024,3,1,1)],False)
        self.classifier=nn.Sequential(nn.Linear(7*7*1024,4096),
                                      nn.Linear(4096,self.output_dim))
    def forward(self,x):

        x=self.layers1.forward(x)
        #print(x.size())
        x = self.layers2.forward(x)
        #print(x.size())
        x = self.layers3.forward(x)
        #print(x.size())
        x = self.layers4.forward(x)
        #print(x.size())
        x = self.layers5.forward(x)
        #print(x.size())
        x = self.layers6.forward(x)
        #print(x.size())
        x=x.view(x.size(0),7*7*1024)
        x=self.classifier(x)
        return x

# the output of the yolo_net:
# SxSx(B*5+C)
# [7,7,0:5],[7,7,5:10],[7,7,10:30]

class darknet19(nn.Module):
    def __init__(self):
        super(darknet19,self).__init__()
        self.layers1=BasicConv2d([(3,32,3,1,1)],True)
        self.layers2=BasicConv2d([(32,64,3,1,1)],True)
        self.layers3=BasicConv2d([(64,128,3,1,1),
                                  (128,64,1,1,0),
                                  (64,128,3,1,1)],True)
        self.layers4=BasicConv2d([(128,256,3,1,1),
                                  (256,128,1,1,0),
                                  (128,256,3,1,1)],True)
        self.layers5=BasicConv2d([(256,512,3,1,1),
                                  (512,256,1,1,0),
                                  (256,512,3,1,1),
                                  (512,256,1,1,0),
                                  (256,512,3,1,1)],True)
        self.layers6=BasicConv2d([(512,1024,3,1,1),
                                  (1024,512,1,1,0),
                                  (512,1024,3,1,1),
                                  (1024,512,1,1,0),
                                  (512,1024,3,1,1)],False)
        self.convs18=nn.Sequential(self.layers1,self.layers2,self.layers3,
                                 self.layers4,self.layers5,self.layers6)
        self.conv19=nn.Conv2d(1024,1000,1,1,0)
    def forward(self,x):

        x=self.layers1.forward(x)
        #print(x.size())
        x = self.layers2.forward(x)
        #print(x.size())
        x = self.layers3.forward(x)
        #print(x.size())
        x = self.layers4.forward(x)
        #print(x.size())
        x = self.layers5.forward(x)
        #print(x.size())
        x = self.layers6.forward(x)
        #print(x.size())

        #x=self.convs18(x)
        #print(x.size())

        ##### the 19th conv2d layer #####
        x=self.conv19.forward(x)
        #print(x.size())
        ##### avgpooling layer #####
        x=nn.AvgPool2d(kernel_size=x.size(-1)).forward(x)
        #print(x.size())
        x=x.squeeze(-1).squeeze(-1)
        #print(x.size())
        ##### softmax layer #####
        x=nn.Softmax().forward(x)
        #print(x.size())
        return x

class yolo2(nn.Module):
    def __init__(self,darknet19):
        super(yolo2,self).__init__()
        self.convs18=darknet19.convs18






if __name__ == '__main__':
    yolo=yolo(S,B,C)
    darknet19=darknet19()
    if torch.cuda.is_available():
        yolo.cuda()
        darknet19.cuda()
    input_yolo=Variable(Tensor(np.zeros((20,3,448,448))))
    input_dark=Variable(Tensor(np.zeros((32,3,224,224))))
    print(darknet19.forward(input_dark))