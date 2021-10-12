import numpy as np
import torch 
import torch.nn as nn 


class Process_MOT():
    """
    Returns annotation in the following format 
    (bb, classe)

    """
    def __init__(self,anno_file):
        self.anno_file = anno_file

    def read_file(self):
        f = open(self.anno_file,"r")
        lines = f.readlines()
        return lines
    
    def process_line(self,line):
        line = line.replace('\n','').split(',')
        line = [int(i) for i in line[:-1]]+ [float(line[-1])]
        frame, ids, x1, y1, x2,y2, confidence, classe , visibility= line
        x1 = x1-1
        y1 = y1 -1
        x2 = x1+ x2-1
        y2 = y1 + y2-1
        return frame, ids, x1, y1, x2,y2, confidence, classe , visibility

    def return_anno_for_frame(self,frame_n):
        """
        Returns the annotation for the frame_n frame
        """
        f = self.read_file()
        list_of_anno = []
        for line in f :
          frame, ids, x1, y1, x2,y2, confidence, classe , visibility = self.process_line(line)
          if int(frame)== frame_n:
              list_of_anno.append(([x1,y1,x2,y2],classe,ids)) 
        return list_of_anno



    def return_consecutive_frame_anno(self,frame_n,frame_m):
        """
        Returns the displacement between frame_n and frame_m

        Args:
            frame_n ([type]): [description]
            frame_m ([type]): [description]

        Returns:
            [list_of_displacement]: (displacement, class , id) of objects in frame
        """
        frame_n_list = self.return_anno_for_frame(frame_n)
        frame_m_list = self.return_anno_for_frame(frame_m)
        displacement = []
        ids = [i[2] for i in frame_n_list]
        not_unique = [i[2] for i in frame_n_list for j in frame_m_list if i[2]==j[2] not in frame_m_list]
        unique = list(set(ids)-set(not_unique))
        for bb_n, classe_n,id_n in frame_n_list:
            for  bb_m, classe_m,id_m in frame_m_list:
                if id_m== id_n and classe_n==1==classe_m:
                    displacement.append((list(np.array(bb_n)- np.array(bb_m)),id_n))
        for id_n in unique:
            displacement.append(([0,0,0,0], id_n))
        return displacement



class Process_BB(nn.Module):
    """
    Class to process every Bounding Box
    For the moment the bb input is of (top left , bottom right)
    """
    def __init__(self,bb):
      super(Process_BB,self).__init__()
      self.x1, self.y1, self.x2, self.y2 = bb

    def HW(self):
      """
      returns Height and Width of image

      """
      return (self.x1 + self.x2, self.y1 + self.y2)

    def coco_bb(self):
      """
      Coco annotation : Top Left Corner and W,H
      """
      return (self.x1, self.y1, self.x2+self.x1, self.y1 + self.y2)
