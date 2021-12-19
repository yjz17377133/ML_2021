from backbone import Resnet
from torch import nn
import torch

class BaseCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseCNNModel, self).__init__()
        self.num_classes = num_classes

        x1_inchannel = 64
        self.x1_inchannel = x1_inchannel

        self.x1_classify_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(x1_inchannel, num_classes, kernel_size=1)
        )

        self.up1 = nn.Upsample(56, mode='bilinear', align_corners=False)

        self.conv1_up = nn.Sequential(
            nn.Conv2d(2048, x1_inchannel, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, x1_inchannel, kernel_size=1),
            nn.ReLU()
        )

        self.x1_head_arr = self.classify(self.num_classes, x1_inchannel)

    def classify(self, num, inchannel):
        lst = nn.ModuleList()
        for i in range(num):
            lst.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(inchannel, 1, kernel_size=1)
                )
        )
        return lst

    def part_res(self, inp, model, num):
        outs = []
        for i in range(num):
            outs.append(model[i](inp[:, i, :, :, :]).squeeze(dim=-1).squeeze(dim=-1))
        outs = torch.cat(outs, dim=-1)
        return outs
    
    def forward(self, x1, x4):
        x1 = self.conv2(x1)
        x1 = x1 + self.up1(self.conv1_up(x4))

        att_1 = self.x1_classify_attention(x1)

        att_1 = att_1.unsqueeze(dim=2)
        x1 = x1.unsqueeze(dim=1)
        
        x1_out = x1 * att_1

        x1_out = self.part_res(x1_out, self.x1_head_arr, self.num_classes)
        return x1_out

class BaseNet(nn.Module):
    def __init__(self, num_classes):
        super(BaseNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = Resnet()
        self.cnn_part = BaseCNNModel(num_classes)
    
    def forward(self, inp):
        #
        x0, x1, x2, x3, x4 = self.backbone(inp)
        x1_out = self.cnn_part(x1, x4)

        #x4_out = self.classify_4(x4_out).squeeze(dim=-1).squeeze(dim=-1)
        return x1_out
