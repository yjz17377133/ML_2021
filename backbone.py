import torch
from torch import nn
from torchvision import models


class Resnet101(nn.Module):

    def __init__(self):
        super(Resnet101, self).__init__()
        module = models.resnet101(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.bn1,
            module.relu,
        )
        self.maxpool = module.maxpool
        self.layer1 = module.layer1
        self.layer2 = module.layer2 # nn.Sequential(module.layer2)
        self.layer3 = module.layer3 # nn.Sequential(module.layer3)
        self.layer4 = module.layer4 #nn.Sequential(module.layer4)
        self.avgpool = nn.Sequential(module.avgpool)

    def forward(self, inp):
        x = self.features(inp) # [4, 64, 112, 112]
        lmx = self.maxpool(x)# [4, 256, 56, 56]
        l1 = self.layer1(lmx)# [4, 256, 56, 56]
        l2 = self.layer2(l1) # [4, 512, 28, 28]
        l3 = self.layer3(l2) # [4, 1024, 14, 14]
        l4 = self.layer4(l3) # [4, 2048, 7, 7]
        # x = self.avgpool(x)  # (batch,2048,1,1)
        return x, l1, l2, l3, l4

class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50, self).__init__()
        module = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.bn1,
            module.relu,
        )
        self.maxpool = module.maxpool
        self.layer1 = module.layer1
        self.layer2 = module.layer2 # nn.Sequential(module.layer2)
        self.layer3 = module.layer3 # nn.Sequential(module.layer3)
        self.layer4 = module.layer4 #nn.Sequential(module.layer4)
        self.avgpool = nn.Sequential(module.avgpool)

    def forward(self, inp):
        x = self.features(inp) # [4, 64, 112, 112]
        lmx = self.maxpool(x)# [4, 256, 56, 56]
        l1 = self.layer1(lmx)# [4, 256, 56, 56]
        l2 = self.layer2(l1) # [4, 512, 28, 28]
        l3 = self.layer3(l2) # [4, 1024, 14, 14]
        l4 = self.layer4(l3) # [4, 2048, 7, 7]
        # x = self.avgpool(x)  # (batch,2048,1,1)
        return x, l1, l2, l3, l4

class Resnet34(nn.Module):

    def __init__(self):
        super(Resnet34, self).__init__()
        module = models.resnet34(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.bn1,
            module.relu,
        )
        self.maxpool = module.maxpool
        self.layer1 = module.layer1
        self.layer2 = module.layer2 # nn.Sequential(module.layer2)
        self.layer3 = module.layer3 # nn.Sequential(module.layer3)
        self.layer4 = module.layer4 #nn.Sequential(module.layer4)
        self.avgpool = nn.Sequential(module.avgpool)

    def forward(self, inp):
        x = self.features(inp) # [4, 64, 112, 112]
        lmx = self.maxpool(x)# [4, 256, 56, 56]
        l1 = self.layer1(lmx)# [4, 256, 56, 56]
        l2 = self.layer2(l1) # [4, 512, 28, 28]
        l3 = self.layer3(l2) # [4, 1024, 14, 14]
        l4 = self.layer4(l3) # [4, 2048, 7, 7]
        # x = self.avgpool(x)  # (batch,2048,1,1)
        return x, l1, l2, l3, l4

class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()
        #module = models.resnet101(pretrained=True)
        module = models.resnet18(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.bn1,
            module.relu,
        )
        self.maxpool = module.maxpool
        self.layer1 = module.layer1
        self.layer2 = module.layer2 # nn.Sequential(module.layer2)
        self.layer3 = module.layer3 # nn.Sequential(module.layer3)
        self.layer4 = module.layer4 #nn.Sequential(module.layer4)
        self.avgpool = nn.Sequential(module.avgpool)

    def forward(self, inp):
        x = self.features(inp) # [4, 64, 112, 112]
        lmx = self.maxpool(x)# [4, 256, 56, 56]
        l1 = self.layer1(lmx)# [4, 256, 56, 56]
        l2 = self.layer2(l1) # [4, 512, 28, 28]
        l3 = self.layer3(l2) # [4, 1024, 14, 14]
        l4 = self.layer4(l3) # [4, 2048, 7, 7]
        # x = self.avgpool(x)  # (batch,2048,1,1)
        return x, l1, l2, l3, l4

if __name__ =='__main__':
    net = Resnet()
    tmp_data = torch.randn(4,3,224,224)
    net = net.cuda()
    tmp_data=tmp_data.cuda()
    s = net(tmp_data)
    print('hello')