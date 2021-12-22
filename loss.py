import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class ClassifyLoss(nn.Module):
    def __init__(self, num_classes):
        nn.Module.__init__(self)
        self.num_class = num_classes
    
    def forward(self, class_vec, classes):
        classes = torch.nn.functional.one_hot(classes, self.num_class)
        epsilon = 1e-7
        class_vec = F.log_softmax(class_vec,dim=-1)
        loss = -1 * classes * class_vec
        # loss = torch.mean(loss)
        loss = loss.sum(1).mean(0)
        # print("yjz_debug:classify",loss)
        return loss