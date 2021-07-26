import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchsummary import summary 

class YOLOv1(nn.Module):
    def __init__(self, input_channels=3, grid = 7, num_boxes=2):
        super(YOLOv1, self).__init__()
        self.model_name = 'YOLOv1_Original'
        self.input_channels = input_channels
        self.grid = grid
        self.num_boxes = num_boxes

        # The model
        self.darknet = nn.Sequential(
        self.conv_block(self.input_channels, 512,7,2,3)
        )


    def conv_block(self, in_c, out_c, kernel_, stride_, padding_):
        return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_, stride_, padding_, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self,x):
        return self.darknet(x)

model = YOLOv1()
print(summary(model, (3,512,512)))
