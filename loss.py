 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_k(nn.Module):
    """
    Pixel-wise Logistic Regression with focal loss
    Paper : https://arxiv.org/pdf/1904.07850.pdf Preliminary
    """
    def __init__(self):
        super(Loss_k,self).__init__()
        self.beta = 4
        self.alpha = 2

    def get_one_or_others(self,gt):
        pos_indices = gt.eq(1).float()
        neg_indices = gt.lt(1).float()
        return pos_indices, neg_indices

    def get_beta_loss_term(self,gt):
        return torch.pow(1-gt,self.beta)

    def get_alpha_loss_term(self,pred):
        return torch.pow(1-pred,self.alpha)

    def forward(self,pred,gt):

        print(torch.max(pred))
        pos_indices, neg_indices = self.get_one_or_others(gt)
        print(torch.unique(pos_indices))
        print(torch.unique(neg_indices))
        pos_loss = self.get_alpha_loss_term(pred)*torch.log(pred)*pos_indices
        neg_loss = self.get_beta_loss_term(gt)*torch.pow(pred,self.alpha)*torch.log(1-pred)*neg_indices

        number_of_keypoints = torch.sum(pos_indices)
        pos_loss = torch.sum(pos_loss)
        neg_loss = torch.sum(neg_loss)

        if number_of_keypoints == 0 :
            return -neg_loss
        else : 
            return -(neg_loss + pos_loss)/(number_of_keypoints)

class Loss_Offset(nn.Module):
    """
    Offset Loss using smoothL1 instead of L1
            Prediction of offset of prediction in R²
            Gt is offset in R² (keypoint/R - round(keypoint/R)). 
    returns:
    """
    def __init__(self,R = 4):
          super(Loss_Offset,self).__init__()
          self.R = R

    def forward(self,offset_pred,keypoints):
        gt = torch.div(keypoints,self.R)- torch.round(torch.div(keypoints,self.R))
        N = gt.shape[0]
        l1 = F.smooth_l1_loss(offset_pred,gt)
        return l1/N

class Loss_Size(nn.Module):
    """
    Size Loss : Takes input : 
        Prediction of size of object in R²
        Gt is objectg size in R² (x1-x2, y2-y1). 
    returns:
        Size Loss
    """
    def __init__(self):
        super(Loss_Size,self).__init__()


    def forward(self,pred, gt):

        l1 = F.smooth_l1_loss(pred,gt)
        N = gt.shape[0]
        return l1/N


class Loss_Displacement(nn.Module):
    """
    Size Loss : Takes input : 
        Prediction of size of object in R²
        Gt is objectg size in R² (x1-x2, y2-y1). 
    returns:
        Size Loss
    """
    def __init__(self):
        super(Loss_Displacement,self).__init__()


    def forward(self,pred, gt):

        l1 = F.smooth_l1_loss(pred,gt)
        N = gt.shape[0]
        return l1/N


class Loss_overall(nn.Module):
    """
    Computes the overall loss for training CenterNet
    """
    def __init__(self,R = 4):
        super(Loss_overall,self).__init__()
        self.R = R
        self.lambda_size = 0.1
        self.lambda_offset = 1

    def forward(self,pred_k,pred_offset, pred_size,pred_displacement, gt_k,gt_offset,gt_size,gt_displacement):
        L_k = Loss_k()(pred_k,gt_k)
        L_size= Loss_Size()(pred_size,gt_size)
        L_offset = Loss_Offset()(pred_offset,gt_offset)
        L_displacement = Loss_Displacement()(pred_displacement,gt_displacement)
        return L_k+ self.lambda_size*L_size + self.lambda_offset*L_offset + L_displacement

