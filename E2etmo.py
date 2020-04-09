import torch
import torch.nn as nn
from torch.nn import init
from Gdn import Gdn2d, Gdn1d
from torch.autograd import Variable
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager, Lock
import threading

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.uniform_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)

    # elif classname.find('BatchNorm2d') != -1:
    #     init.constant_(m.weight.data, 1.0)
    #     # init.constant_(m.bias.data,   0.0)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]),requires_grad=True)

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001,affine=False)

    def forward(self, x):
        return self.w_0 * self.bn(x)

class lrelu(nn.Module):
    def __init__(self):
        super(lrelu, self).__init__()
    def forward(self, x):
        return torch.max(x*0.2, x)



def build_net(norm=AdaptiveNorm, layer=5, width=32):
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(width),
                   lrelu(),
                   ]

    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        lrelu(),
        nn.Conv2d(width,  1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init)

    return net



class E2ETMO(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self,layer):
        super(E2EPOIR, self).__init__()
        self.cnn = build_net(layer=4)
        self.luminance_cnn = build_net(layer=4)
        self.d_max = 300
        self.d_min = 5


    def forward(self, x):

        nlev = x.__len__()
        y = [0] * nlev
        z = [0] * nlev

        # for
        for nlp_index in range(nlev-1):
        #    start = time.time()
            z[nlp_index] = self.cnn(x[nlp_index])

        z[nlev - 1] = self.luminance_cnn(x[nlev - 1])

        result = self.reconstract_nlp(z)

        result = self.Constraints(result)
        return result

    def init_model(self, path):
        self.cnn.load_state_dict(torch.load(path))

    def Constraints(self, result):
        result = torch.sigmoid(result) # [0, 1]

        result = (self.d_max - self.d_min) * result + self.d_min

        return result

    def reconstract_nlp(self, pyr):
        if pyr[0].is_cuda:
            filt = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]], dtype=torch.float,device=pyr[0].get_device()).unsqueeze(0).unsqueeze(0)
        else:
            filt = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                 [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                 [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                 [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                 [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]], dtype=torch.float).unsqueeze(0).unsqueeze(0)


        nlev = pyr.__len__()
        # for i in range(nlev):
        #     pyr[i] = pyr[i].unsqueeze(0)
        R = pyr[nlev - 1]
        for index in range(pyr.__len__() - 2, -1, -1):
            h_odd = R.shape[2] * 2 - pyr[index].shape[2]
            w_odd = R.shape[3] * 2 - pyr[index].shape[3]
            R = pyr[index] + self.upsample(R, [h_odd, w_odd], filt)
        return R

    def upsample(self, img, odd, filt):
        img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        h = 2 * img.shape[2]
        w = 2 * img.shape[3]
        if img.is_cuda:
            o = torch.zeros([img.shape[0], img.shape[1], h, w], device=img.get_device())
        else:
            o = torch.zeros([img.shape[0], img.shape[1], h, w])
        o[:, :, 0:h:2, 0:w:2] = 4 * img
        o = F.conv2d(o, filt, padding=math.floor(filt.shape[2] / 2))
        o = o[:, :, 2:h - 2 - odd[0], 2:w - 2 - odd[1]]
        return o

    def zero_padding(self,nlp_list):
        length = len(nlp_list)
        z = [0]*length
        w = nlp_list[0].size(2)
        h = nlp_list[0].size(3)
        for i in range(length-2,0,-1):
            z[i] = F.pad(nlp_list[i],(math.floor((h - nlp_list[i].size(3))/2),math.ceil((h - nlp_list[i].size(3))/2),math.floor((w - nlp_list[i].size(2))/2),math.floor((w - nlp_list[i].size(2))/2)),mode='constant',value=0)
        k = torch.cat([nlp_list[0],z[1],z[2],z[3]],dim=0)
        return k











