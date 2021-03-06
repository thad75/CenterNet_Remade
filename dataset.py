from torch.utils.data import Dataset, DataLoader
from process import * 
from heatmap_process import * 
from os import listdir
import os
import cv2 
from torchvision import transforms

class CenterDataset_MOT(Dataset):
    def __init__(self,folder_of_images,anno_file, transform = None, R=4):
      self.anno_file = Process_MOT(anno_file)
      self.folder_of_images = sorted([folder_of_images + '/' + i for i in os.listdir(folder_of_images)])
      self.R = 4
      self.transform = transform

    def downsize_map(self,image):  
          pipeline = nn.ModuleList([nn.MaxPool2d((2,2)) for i in range(self.R//2)])
          for i, l in enumerate(pipeline):
              image =pipeline[i](image) 
          return image

    def __getitem__(self,idx):
      if idx ==0 : 
        idx = 1
      annotation = self.anno_file.return_anno_for_frame(idx)
      BB = [i[0] for i in annotation]
      classes = max([i[1] for i in annotation])+1
      image = torch.from_numpy(cv2.imread(self.folder_of_images[idx])).permute(2,1,0)
      init_shape = image.shape[1], image.shape[2]        
      if self.transform:
        image = self.transform(image)
      out_shape = image.shape[1], image.shape[2]
      if idx == 1:
        prev_image = self.folder_of_images[idx]
        prev_image_heatmap = Create_Heatmap_GT(torch.zeros((classes,image.shape[1],image.shape[2]))).create_keypoint_map(self.anno_file.return_anno_for_frame(idx),init_shape, out_shape, wanted_class = 1)

        if self.transform : 
            
            prev_image_heatmap = self.transform(prev_image_heatmap)
            prev_image = torch.from_numpy(cv2.imread(prev_image)).permute(2,1,0)
            prev_image = self.transform(prev_image)
            
            
        else : 
            prev_image = torch.from_numpy(cv2.imread(prev_image)).permute(2,1,0)

        displacement = self.anno_file.return_consecutive_frame_anno(idx,idx)
        # annotation_prev = self.anno_file.return_anno_for_frame(idx)

      else :
        prev_image = self.folder_of_images[idx-1]
        prev_image_heatmap = Create_Heatmap_GT(torch.zeros((classes,image.shape[1],image.shape[2]))).create_keypoint_map(self.anno_file.return_anno_for_frame(idx-1), init_shape,out_shape,wanted_class = 1)
        

        if self.transform : 
            
            prev_image_heatmap = self.transform(prev_image_heatmap)
            prev_image = torch.from_numpy(cv2.imread(prev_image)).permute(2,1,0)
            prev_image = self.transform(prev_image)
        else : 
            prev_image = torch.from_numpy(cv2.imread(prev_image)).permute(2,1,0)

        # annotation_prev = self.anno_file.return_anno_for_frame(idx-1)
        displacement = self.anno_file.return_consecutive_frame_anno(idx,idx-1)
        
      # print(displacement)
      BB_prev = [i[0] for i in displacement]

      downsized_keypoint_heatmap = self.downsize_map(Create_Heatmap_GT(torch.zeros((classes,image.shape[1],image.shape[2]))).create_keypoint_map(annotation,init_shape, out_shape,wanted_class = 1))
      downsized_size_map = self.downsize_map(Create_Heatmap_GT(torch.zeros((2,image.shape[2],image.shape[1]))).create_size_map(BB,init_shape, out_shape))
      downsized_offset_map = self.downsize_map(Create_Heatmap_GT(torch.zeros((2,image.shape[1],image.shape[2]))).create_offset_map(BB,init_shape, out_shape))     
      downsized_displacement_map = self.downsize_map(Create_Heatmap_GT(torch.zeros((2,image.shape[1],image.shape[2]))).create_displacement_map(BB,BB_prev,init_shape, out_shape))


      return {'current_image': image.float(),
              'previous_frame': prev_image.float(),
              'prev_image_heatmap':prev_image_heatmap.float(),
              'downsized_keypoint_heatmap':downsized_keypoint_heatmap.float(),
              'downsized_size_map': downsized_size_map.permute(0,2,1).float(),
              'downsized_offset_map': downsized_offset_map.float(),
              'downsized_displacement_map': downsized_displacement_map.float()}

    def __len__(self):
      return len(self.folder_of_images)



