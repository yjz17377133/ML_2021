import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class ClassifyLoss(nn.Module):
    def __init__(self, num_classes):
        nn.Module.__init__(self)
        self.num_class = 196
    
    def forward(self, class_vec, classes):
        classes = torch.nn.functional.one_hot(classes, self.num_class)
        epsilon = 1e-7
        class_vec = torch.sigmoid(class_vec)
        loss = -classes * torch.log(class_vec + epsilon) - (1.0 - classes) * torch.log((1.0 - class_vec) + epsilon)
        loss = torch.mean(loss)
        print("yjz_debug:classify",loss)
        return loss