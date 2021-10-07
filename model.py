from backbone import *
from head import *
from loss import * 
from torchvision import transforms
import pytorch_lightning as pl

class Pre_Backbone(nn.Module):
    def __init__(self,in_channels, out_channels):  
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_backbone_layer = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, 
                      out_channels =  self.out_channels,
                      stride = 1, padding = 3, bias = False),
            nn.BatchNormd2d(2),
            nn.ReLU(inplace= True)
        )
    
    def forward(self,x):
        return self.pre_backbone_layer(x)



class CenterNet(pl.LightningModule):
    def __init__(self,):
        super(CenterNet,self).__init__()
        self.pre_backbone_layer = Pre_Backbone(3,128)
        self.pre_backbone_layer_heatmap = Pre_Backbone(1,128)
        self.backbone = nn.Sequential(
            Hourglass(2,128 ),
            nn.MaxPool2d(2,2),
            nn.ReLU())
        self.head = nn.ModuleList()
        self.head.add_module('detection', Detection_Head(128))
        self.head.add_module('offset', Offset_Head(128))
        self.head.add_module('size', Size_Head(128))
        self.head.add_module('displacement', Size_Head(128))

    def forward(self,x,prev_x,heatmap_prev_x):
        x = self.pre_backbone_layer(x)
        if prev_x is not None:
            x += self.pre_backbone_layer(prev_x)
        if heatmap_prev_x is not None:
            x + self.pre_backbone_layer_heatmap(heatmap_prev_x)
        feature = self.backbone(x)
        del x
        offset_map = self.head.offset(feature)
        detection_map = self.head.detection(feature)
        size_map = self.head.size(feature)
        displacement_map = self.headLdisplacement(feature)
        return offset_map, detection_map, size_map , displacement_map


    def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer


    def training_step(self, train_batch, batch_idx):
		actual_frame,previous_frame, tracked_objects  =train_batch['current_frame'],train_batch['previous_frame'],train_batch['previous_frame_heatmap']
		offset_map, detection_map, size_map , displacement_map = self.forward(actual_frame,previous_frame, tracked_objects)

    def validation_step(self, val_batch, batch_idx):
		actual_frame,previous_frame, tracked_objects = train_batch['current_frame'],train_batch['previous_frame'],train_batch['previous_frame_heatmap']
		offset_map, detection_map, size_map , displacement_map = self.forward(actual_frame,previous_frame, tracked_objects)
