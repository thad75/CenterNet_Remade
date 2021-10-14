import torch
import torch.nn as nn


class Decode_Map(nn.Module):

        """
        This class is used to decode the HM returned by the model.
        Each map should be BatchSize,Channels,H,W
        
        """
        def __init__(self,heatmap_detection, heatmap_size, heatmap_offset, heatmap_displacement):
                self.heatmap_detection = heatmap_detection 
                self.heatmap_size = heatmap_size
                self.heatmap_offset = heatmap_offset
                self.heatmap_displacement = heatmap_displacement

    
        def top_k_channels(self,maps,k=100):
            BS,C,H,W = maps.size()
            top_k_scores, top_k_indices = torch.topk(maps.view(BS,C,-1),k)
            top_k_indices = top_k_indices % (H*W)
            top_k_ys = (top_k_indices / W).int().float()
            top_k_xs = (top_k_indices % W).int().float()
            return top_k_scores, top_k_indices, top_k_ys, top_k_xs, BS

        def get_feature(self,feature, indices):
            dim = feature.size(2)
            indices = indices.unsqueeze(2).expand(indices.size(0),indices.size(1), dim)
            return feature.gather(1,indices)

        def top_k(self,maps,k=100):
            top_k_scores, top_k_indices, top_k_ys, top_k_xs, BS= self.top_k_channels(maps,k)
            top_k_score , top_k_indice = torch.topk(top_k_scores.view(BS, -1), k)
            top_k_classe = (top_k_indice/k).int()
            top_k_indices = self.get_feature(top_k_indices.view(BS,-1,1), top_k_indice).view(BS,k)
            top_k_ys = self.get_feature(top_k_ys.view(BS,-1,1), top_k_indice).view(BS,k)
            top_k_xs = self.get_feature(top_k_xs.view(BS,-1,1), top_k_indice).view(BS,k)
            return top_k_scores, top_k_indices,top_k_classe, top_k_ys, top_k_xs
        

        def nms(self,heatmap, kernel = 3):

            """
            Retrieves the max values of the heatmap
            ##review
            """
            pad = (kernel-1)//2
            hmax = nn.functional.max_pool2d(heatmap, (kernel,kernel), stride = 1, padding= pad)
            keep = (hmax==heatmap).float()
            return heatmap*keep


        def decode(self,heatmap,k=100):
            heatmap = self.nms(heatmap)
            top_k_scores, top_k_indices,top_k_classe, top_k_ys, top_k_xs = self.top_k(heatmap)
            

             