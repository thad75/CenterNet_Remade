import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from process import Process_BB
import itertools
class Process_Heatmap_GT(nn.Module):

    """
    Create the GT HeatMap for object detection
    """
    def __init__(self,heatmap,R=4):
        super(Process_Heatmap_GT,self).__init__()
        self.heatmap = heatmap
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
        gaussian_kernel = gaussian_kernel/torch.max(gaussian_kernel)

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
            left_c,right_c,top_c,bottom_c = tuple(map(lambda i, j: i - j, (left,right,top,bottom ), tuple_of_coord))
            gaussian = gaussian[0+left_c: W+ right_c,0+top_c: H+ bottom_c]
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
        x1,y1,x2,y2 = bounding_box
        self.heatmap[0,y1,x1:x2]=1
        self.heatmap[0,y1:y2,x1]=1
        self.heatmap[0,y2,x1:x2]=1
        self.heatmap[0,y1:y2,x2]=1
        return self.heatmap

    def draw_size(self, bounding_box):
        x1,y1,x2,y2 = bounding_box
        self.heatmap[0,(y1+y2)//2,x1:x2]=int(abs(y2-y1)) #1
        self.heatmap[0,y1:y2,(x1+x2)//2]=int(abs(x2-x1))#1
        return self.heatmap

    def draw_offset(self,bounding_box):
        x1,y1,x2,y2 = bounding_box
        p = torch.Tensor([(x1+x2)/2/self.R, (y1+y2)/2/self.R])
        p_int = torch.floor(p)
        gt_p = p-p_int
        self.heatmap[0,int(p[0]+gt_p[0])*self.R,int(p[1])*self.R]=gt_p[0] #1
        self.heatmap[1,int(p[0])*self.R,int(p[1]+gt_p[1])*self.R]=gt_p[1] #1   
        return self.heatmap

    def draw_displacement(self,bounding_box,displacement):
        x1,y1,x2,y2 = bounding_box
        x1_d,y1_d,x2_d,y2_d = displacement
        p = torch.Tensor([(x1+x2)/2, (y1+y2)/2])
        self.heatmap[0,int(p[0]+x1_d),int(p[1])]=1
        self.heatmap[0,int(p[0]+x2_d),int(p[1])]=1
        self.heatmap[1,int(p[0]),int(p[1]+y1_d)]=1  
        self.heatmap[1,int(p[0]),int(p[1]+y2_d)]=1        
    
        return self.heatmap

class Create_Heatmap_GT(nn.Module):
    """
    Creates a the Heatmap depending of a list of tuple (BB, class)
    """
     
    def __init__(self, heatmap):
        super(Create_Heatmap_GT,self).__init__()
        # self.list_of_bounding_boxes = list_of_bounding_boxes
        self.heatmap = heatmap

    def create_keypoint_map(self,list_of_bounding_boxes, input_shape, output_shape,wanted_class = None):
        for bb, classe,_ in list_of_bounding_boxes:  
            bb = self.convert_size( bb , input_shape, output_shape)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_gaussian(bb,classe)
        if wanted_class: 
            return self.heatmap[wanted_class,:,:].unsqueeze(0)
        return self.heatmap

    def create_size_map(self,list_of_bounding_boxes, input_shape, output_shape):
        for bb in list_of_bounding_boxes:
            bb = self.convert_size( bb , input_shape, output_shape)

            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_size(bb)
        return self.heatmap

    def create_offset_map(self,list_of_bounding_boxes, input_shape, output_shape):
        for bb in list_of_bounding_boxes:
            bb = self.convert_size( bb , input_shape, output_shape)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_offset(bb)
        return self.heatmap

    def create_rectangle_map(self,list_of_bounding_boxes, input_shape, output_shape):
        for bb in list_of_bounding_boxes:
            bb = self.convert_size( bb , input_shape, output_shape)
            # print(bb)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_bb(bb)
        return self.heatmap

    def create_displacement_map(self,list_of_bounding_boxes, list_of_bounding_boxes_prev,input_shape, output_shape):
        for i,j in itertools.zip_longest(list_of_bounding_boxes,list_of_bounding_boxes_prev):
            # print(i,j)
            if i is None:
              i = [0,0,0,0]
            if j is None:
              j = [0,0,0,0]
            i = self.convert_size( i , input_shape, output_shape)            
            j = self.convert_size( j , input_shape, output_shape)
            self.heatmap = Process_Heatmap_GT(self.heatmap).draw_displacement(i,j)
        return self.heatmap


    def convert_size(self,bb, input_shape,output_shape):
   
            W_init, H_init= input_shape
            W_out, H_out= output_shape
            W,H  = W_out/W_init, H_out/H_init
            x1,y1,x2,y2 = bb
            x1,y1,x2,y2 = x1*W,y1*H,x2*W,y2*H
            bb = [x1,y1,x2,y2]
            return [int(i) for i in bb]
        
   