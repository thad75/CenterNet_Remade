import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from process import Process_BB

class Process_Heatmap_GT(nn.Module):

    """
    Create the GT HeatMap for object detection
    """
    def __init__(self,heatmap,R=4):
        super(Process_Heatmap_GT,self).__init__()
        self.heatmap = heatmap
        print(self.heatmap.shape)
        self.R = 4

    def radius(self,bounding_box):
        """
        as explained in https://arxiv.org/pdf/1808.01244.pdf
        We want values that could be in a circle within the BB in which every
        point has min 0.3 with the ground truth

        Must be verified before testing
        """
        pi = torch.Tensor([math.pi])
        H,W = Process_BB(bounding_box).HW()
        r = torch.sqrt(0.3*H*W/pi)
        sigma = (1/3)*r
        return sigma

    def convert_BB_to_keypoint(self,bounding_box):
        """
        Converts a BB to Keypoint 
        Keypoint correspond to Center of Image or Joints 
        """
        x1,y1,x2,y2 = bounding_box
        return (int((x1+x2)/2), int((y1+y2)/2))

    def convert_BB_to_size(self,bounding_box):
        x1,y1,x2,y2 = bounding_box

        return (abs(x1-x2),abs(y1-y2))

    def get_gaussian(self,bounding_box,sigma):
        """
        Creates a map that contains the Peak of the gaussian at the keypoint
        and returns the map corresponding 

        """
        shape = bounding_box
        m,n = [int((ss - 1)/2) for ss in shape]
        x,y = torch.range(-m,m,1).view(1,-1),torch.range(-n,n,1).view(-1,1)
        gaussian_kernel =torch.transpose(torch.exp(-(torch.pow(x,2)+ torch.pow(y,2)))/2*sigma*sigma,0,1)
        # gaussian_kernel = gaussian_kernel/torch.max(gaussian_kernel)
        print(gaussian_kernel.shape)
        print(gaussian_kernel)
        return gaussian_kernel

    def process_gaussian_coord(self,tuple_of_coord,gaussian):
        """
        Process the coordinates of the Gaussian to paste on the heatmap
        Border Effect are

        """
        left,right,top,bottom = tuple_of_coord+ ()
        W,H = gaussian.shape
        if left < 0 :
            left = 0
        if right> self.heatmap.shape[1]:
            right = self.heatmap.shape[1]
        if top < 0:
            top = 0
        if bottom > self.heatmap.shape[2]:
            bottom = self.heatmap.shape[2]
        if (left,right,top,bottom ) != tuple_of_coord:
            left_c,right_c,top_c,bottom_c = (tuple(map(lambda i, j: i - j, (left,right,top,bottom ), tuple_of_coord)))
            gaussian = gaussian[left_c: W+ right_c,top_c: H+ bottom_c]
        return left,right,top,bottom, gaussian

    def draw_gaussian(self,bounding_box,classe,min_radius = 60):
        """
        Returns the updated main heatmap with the peak corresponding
        """
        W,H = self.heatmap.shape[1::]
        x,y = self.convert_BB_to_keypoint(bounding_box)
        sigma = self.radius(bounding_box)
        gaussian = self.get_gaussian([2*min_radius,2*min_radius],sigma)
        W_g, H_g = gaussian.shape
        left,right,top,bottom, gaussian = self.process_gaussian_coord((x-W_g//2,x+W_g//2+1,y-H_g//2,y+H_g//2+1), gaussian)
        self.heatmap[classe,left:right,top:bottom ] = torch.max(gaussian,self.heatmap[classe,left:right,top:bottom ])
        return self.heatmap

    def draw_bb(self, bounding_box):
        print('hi',bounding_box)
        x1,y1,x2,y2 = bounding_box
        self.heatmap[0,y1,x1:x2]=1
        self.heatmap[0,y1:y2,x1]=1
        self.heatmap[0,y2,x1:x2]=1
        self.heatmap[0,y1:y2,x2]=1
        return self.heatmap

    def draw_size(self, bounding_box):
        x1,y1,x2,y2 = bounding_box
        print(x1,y1,x2,y2)
        print('hi',bounding_box)

        self.heatmap[0,(y1+y2)//2,x1:x2]=1
        self.heatmap[0,y1:y2,(x1+x2)//2]=1
        return self.heatmap

    def draw_offset(self,bounding_box):
        x1,y1,x2,y2 = bounding_box
        p = torch.Tensor([(x1+x2)/2/self.R, (y1+y2)/2/self.R])
        # printé(p)
        p_int = torch.floor(p)
        print(p, p_int)
        gt_p = p-p_int
        print(gt_p)
        return self.heatmap


class Create_Heatmap_GT(nn.Module):
    """
    Creates a the Heatmap depending of a list of tuple (BB, class)
    """
     
    def __init__(self, heatmap):
        super(Create_Heatmap_GT,self).__init__()
        # self.list_of_bounding_boxes = list_of_bounding_boxes
        self.heatmap = heatmap

    def create_keypoint_map(self,list_of_bounding_boxes):

        for bb, classe in list_of_bounding_boxes:
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_gaussian(bb,classe)

        return self.heatmap

    def create_size_map(self,list_of_bounding_boxes):
        for bb in list_of_bounding_boxes:
            print(bb)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_size(bb)
        return self.heatmap

    def create_offset_map(self,list_of_bounding_boxes):
        for bb in list_of_bounding_boxes:
            print(bb)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_offset(bb)
        return self.heatmap

    def create_rectangle_map(self,list_of_bounding_boxes):
        for bb in list_of_bounding_boxes:
            print(bb)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_bb(bb)
        return self.heatmap


bounding_box = [50,50,120,120]
a = torch.zeros((2,128,128))      
# f= Process_Heatmap_GT(a)
# a = f.draw_gaussian(bounding_box,1)
bounding_boxes = [([0,0,100,80],2),([80,0,110,50],2),([10,30,70,38],2)]#,(,,([10,30,70,38],2)]
only_bb = [i[0] for i in bounding_boxes]
print(only_bb)
# for bb, classe in bounding_boxes:
#     a = Process_Heatmap_GT(a)
#     a= a.draw_gaussian(bb,classe)
# # print(a.unique())

f = Create_Heatmap_GT(a).create_size_map(only_bb)