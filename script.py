from backbone import *
from head import *
from loss import * 
from dataset import *
import torchvision
from torchvision  import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import glob

class Pre_Backbone(nn.Module):
    def __init__(self,in_channels, out_channels):  
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_backbone_layer = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, 
                      out_channels =  self.out_channels,
                      kernel_size = 7,
                      stride = 1, padding = 3, bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace= False)
        )
    
    def forward(self,x):
        return self.pre_backbone_layer(x)



class CenterNet(pl.LightningModule):
    def __init__(self,):
        super(CenterNet,self).__init__()
        self.channels_of_feature = 128
        self.pre_backbone_layer = Pre_Backbone(3,self.channels_of_feature)
        self.pre_backbone_layer_heatmap = Pre_Backbone(1,self.channels_of_feature)
        self.backbone = nn.Sequential(
            Hourglass(1,self.channels_of_feature),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace= False))
        self.head = nn.ModuleList()
        self.head.add_module('detection', Detection_Head(self.channels_of_feature,number_of_class=1))
        self.head.add_module('offset', Offset_Head(self.channels_of_feature))
        self.head.add_module('size', Size_Head(self.channels_of_feature))
        self.head.add_module('displacement', Displacement_Head(self.channels_of_feature))

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
        displacement_map = self.head.displacement(feature)
        return offset_map, detection_map, size_map , displacement_map


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def grid_it(self,tensor):
        grid = torchvision.utils.make_grid(tensor)
        return grid


    
    def training_step(self, train_batch, batch_idx):
        criterion = Loss_overall()
        actual_frame,previous_frame, tracked_objects  =train_batch['current_image'],train_batch['previous_frame'],train_batch['prev_image_heatmap']
        offset_map, detection_map, size_map , displacement_map = self.forward(actual_frame,previous_frame, tracked_objects)
        gt_heatmap, gt_size_map, gt_offset_map, gt_displacement_map = train_batch['downsized_keypoint_heatmap'],train_batch['downsized_size_map'],train_batch['downsized_offset_map'],train_batch['downsized_displacement_map']
        
        loss = criterion( detection_map.clone(),offset_map.clone(), size_map.clone(), displacement_map.clone(),
                         gt_heatmap,gt_offset_map, gt_size_map,  gt_displacement_map )
        #print(torch.max(offset_map),torch.max(detection_map), torch.max(size_map) , torch.max(displacement_map))

        
        self.log("training", loss,on_step=True,on_epoch=True)
        self.logger.experiment.add_image('offset train gt',self.grid_it(gt_offset_map),batch_idx)
        self.logger.experiment.add_image('detection train gt', self.grid_it(gt_heatmap),batch_idx)
        self.logger.experiment.add_image('size train gt',self.grid_it(gt_size_map),batch_idx)
        self.logger.experiment.add_image('displacement train gt',self.grid_it(gt_displacement_map),batch_idx)
        self.logger.experiment.add_image('offset train',self.grid_it(offset_map),batch_idx)
        self.logger.experiment.add_image('detection train', self.grid_it(detection_map),batch_idx)
        self.logger.experiment.add_image('size train',self.grid_it(size_map),batch_idx)
        self.logger.experiment.add_image('displacement train',self.grid_it(displacement_map),batch_idx)
        return loss
    
        
    def validation_step(self, val_batch, batch_idx):
        criterion = Loss_overall()

        actual_frame,previous_frame, tracked_objects = val_batch['current_image'],val_batch['previous_frame'],val_batch['prev_image_heatmap']
        #print(actual_frame.shape,previous_frame.shape, tracked_objects.shape)
        offset_map, detection_map, size_map , displacement_map = self.forward(actual_frame,previous_frame, tracked_objects)
        gt_offset_map, gt_detection_map, gt_size_map , gt_displacement_map = val_batch['downsized_offset_map'],val_batch['downsized_keypoint_heatmap'],val_batch['downsized_size_map'],val_batch['downsized_displacement_map']
        #print(offset_map.shape, detection_map.shape, size_map.shape , displacement_map.shape, '#' , gt_offset_map.shape, gt_detection_map.shape, gt_size_map.shape , gt_displacement_map.shape)
        loss = criterion( detection_map.clone(),offset_map.clone(), size_map.clone(), displacement_map.clone(), 
                         gt_detection_map,gt_offset_map, gt_size_map,  gt_displacement_map )
        #print(torch.max(offset_map),torch.max(detection_map), torch.max(size_map) , torch.max(displacement_map))

        self.log("validation", loss,on_step=True, on_epoch=True)
        self.logger.experiment.add_image('offset valid',self.grid_it(offset_map),batch_idx)
        self.logger.experiment.add_image('detection valid',self.grid_it(detection_map),batch_idx)
        self.logger.experiment.add_image('size valid',self.grid_it(size_map),batch_idx)
        self.logger.experiment.add_image('displacement valid',self.grid_it(displacement_map),batch_idx)


folder_of_images = "MOT20/train/MOT20-01/img1/"
anno_file = "MOT20/train/MOT20-01/gt/gt.txt"
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((500,300)),
                                transforms.ToTensor()])

Dataset = CenterDataset_MOT(folder_of_images, anno_file,transform)


train_size = int(0.8 * len(Dataset))
test_size = len(Dataset) - train_size
print(train_size,test_size)
dataset_train, dataset_validation =  torch.utils.data.random_split( Dataset, [train_size, test_size])


train_loader = DataLoader(dataset_train, batch_size=11,shuffle= True,drop_last = True)
val_loader = DataLoader(dataset_validation, batch_size=3)

# model
model =CenterNet()

# training
tb_logger = pl_loggers.TensorBoardLogger("logs/")
trainer = pl.Trainer(gpus=-1,max_epochs=50,accelerator='dp',logger=tb_logger,log_every_n_steps=1)
trainer.fit(model, train_loader, val_loader)
    
    