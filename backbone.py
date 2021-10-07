import torch 
import torch.nn as nn

class Convolution_Block(nn.Module):
    """
    [Convolutionnal Block used in every models. Combines Conv2d, BatchNorm and LeakyRelu]

    Args:
        in_channels : number of input channels
        out_channels : number of output channels
        kernel_size : kernel size
        stride : stride 
    """

    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(Convolution_Block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = (self.kernel_size-1)//2
        self.pipeline = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding= (self.padding,self.padding), stride = (self.stride, self.stride)),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.001),
        )

    def forward(self,x):
        out = self.pipeline(x)
        return out

class Residual(nn.Module):

    """
    Residual Conv Layer
    Args:
        in_channels : number of input channels
        out_channels : number of output channels
    """


    def __init__(self,in_channels,out_channels):
        super(Residual,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == out_channels :
            self.need_skip = False
        else :
            self.need_skip = True

        self.skip = nn.Sequential(
            Convolution_Block(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size= 1, stride= 1)
        ) if self.need_skip is True else nn.Sequential()

        self.out1 = nn.Sequential(
            Convolution_Block(self.in_channels, self.out_channels//2, kernel_size=1,stride = 1),
        )
        self.out2  = nn.Sequential(
            Convolution_Block(self.out_channels//2, self.out_channels//2, kernel_size=3,stride = 1),
        )
        self.out3  = nn.Sequential(
            Convolution_Block(self.out_channels//2, self.out_channels, kernel_size=1,stride = 1),
        )
        

    def forward(self,x):  
        residual = self.skip(x)
        x = self.out1(x)
        x = self.out2(x)
        x = self.out3(x)
        return x + residual


class Hourglass(nn.Module):
    """
    Hourglas model using residual neural network

    Args:
        number_of_houglass : number of stacked hourglass
        featuer_size : the size of the feature map
        increase : increase in feature map size
    """



    def __init__(self,number_of_hourglass, feature_size, increase = 0):
        super(Hourglass,self).__init__()
        self.feature_size = feature_size 
        self.number_of_hourglass = number_of_hourglass
        self.increase = increase
        self.out_feature_number = feature_size + increase
        self.high_branch = Residual(self.feature_size, self.feature_size)
        self.low_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2)),
            Residual(self.feature_size,self.out_feature_number)
        )

        if self.number_of_hourglass> 1 :
            self.low_branch_2 =  Hourglass(self.number_of_hourglass-1, self.out_feature_number)
        else : 
            self.low_branch_2 =Residual(self.out_feature_number,self.out_feature_number)
        self.low_branch_3 = Residual(self.out_feature_number, self.feature_size)
        self.high_branch_2 = nn.Upsample(scale_factor=2, mode = 'nearest')


    def forward(self,x):
        up = self.high_branch(x)
        low = self.low_branch(x)
        low = self.low_branch_2(low)
        low = self.low_branch_3(low)
        up2 = self.high_branch_2(low)
        return up + up2