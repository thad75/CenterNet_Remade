import torch 
import torch.nn as nn

class Detection_Head(nn.Module):
    """
    Detection Head for CenterNet

    Args:
        nn ([type]): [description]
    """
    def __init__(self,in_channels, number_of_class=80):
        super(Detection_Head,self).__init__()
        self.in_channels = in_channels
        self.out_channels = number_of_class
        self.pipeline = nn.Sequential(

            nn.Conv2d(in_channels  = self.in_channels, 
                      out_channels = self.out_channels,
                      kernel_size = 1,
                      stride = 1),
            nn.MaxPool2d(2,2),
        )
                    

    def forward(self,x):
        x = self.pipeline(x)
        return x

class Size_Head(nn.Module):
    def __init__(self,in_channels):
        super(Size_Head,self).__init__()
        self.in_channels = in_channels
        self.pipeline = nn.Sequential(

            nn.Conv2d(in_channels  = self.in_channels, 
                      out_channels = 2,
                      kernel_size = 1,
                      stride = 1),
            nn.MaxPool2d(2,2),
        )
                    

    def forward(self,x):
        x = self.pipeline(x)
        return x

class Offset_Head(nn.Module):
    def __init__(self,in_channels):
        super(Offset_Head,self).__init__()
        self.in_channels = in_channels
        self.pipeline = nn.Sequential(

            nn.Conv2d(in_channels  = self.in_channels, 
                      out_channels = 2,
                      kernel_size = 1,
                      stride = 1),
            nn.MaxPool2d(2,2),
        )
                    

    def forward(self,x):
        x = self.pipeline(x)
        return x


class Displacement_Head(nn.Module):
    def __init__(self,in_channels):
        super(Displacement_Head,self).__init__()
        self.in_channels = in_channels
        self.pipeline = nn.Sequential(

            nn.Conv2d(in_channels  = self.in_channels, 
                      out_channels = 2,
                      kernel_size = 1,
                      stride = 1),
            nn.MaxPool2d(2,2),
        )
                    

    def forward(self,x):
        x = self.pipeline(x)
        return x