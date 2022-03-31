# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Conv
from utils.downloads import attempt_download
from utils.general import check_version

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model, sscrDetect):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect or t is sscrDetect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble

    


class CWR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        r = 16
        self.conv_1 = nn.Conv2d(dim, dim // r, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(dim // r, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, feature):
        b, c, _, _ = feature.size()

        gap = F.avg_pool2d(feature, (feature.size(2), feature.size(3)), stride=(feature.size(2), feature.size(3)))
        gmp = F.max_pool2d(feature, (feature.size(2), feature.size(3)), stride=(feature.size(2), feature.size(3)))

        gap = self.conv_1(gap)
        gap = F.relu(gap)
        gap = self.conv_2(gap)

        gmp = self.conv_1(gmp)
        gmp = F.relu(gmp)
        gmp = self.conv_2(gmp)

        x = gap + gmp
        x = torch.sigmoid(x)
        x = feature * x

        return x


class SWR(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_size = 3
        self.conv_1_1 = nn.Conv2d(channels, channels,
                                  kernel_size=(kernel_size, 1),
                                  padding=(kernel_size // 2, 0))
        self.conv_1_2 = nn.Conv2d(channels, channels,
                                  kernel_size=(1, kernel_size),
                                  padding=(0, kernel_size // 2))
        self.conv_2_1 = nn.Conv2d(channels, channels,
                                  kernel_size=(1, kernel_size),
                                  padding=(0, kernel_size // 2))
        self.conv_2_2 = nn.Conv2d(channels, channels,
                                  kernel_size=(kernel_size, 1),
                                  padding=(kernel_size // 2, 0))


    def forward(self, features):
        x_1 = self.conv_1_1(features)
        x_1 = F.relu(x_1)
        x_1 = self.conv_1_2(x_1)
        x_1 = F.relu(x_1)

        x_2 = self.conv_2_1(features)
        x_2 = F.relu(x_2)
        x_2 = self.conv_2_2(x_2)
        x_2 = F.relu(x_2)

        x = x_1 + x_2
        x = torch.sigmoid(x)
        x = features * x

        return x

    
class SSCR(nn.Module):
    def __init__(self, cx):
        super().__init__()
        self.cwr=CWR(cx)
        self.swr=SWR(cx)

    def forward(self,x,f):
        x=x+f
        x=self.cwr(x)*self.swr(x)
        x=x+f

        return x


# class SSCR(nn.Module):
#     def __init__(self, cx):
#         super().__init__()
#         self.cwr=CWR(cx)
#         self.swr=SWR(cx)
#         self.conv = nn.Conv2d(cx*2, cx,kernel_size=1, stride=1)

#     def forward(self,x,f):
#         x=torch.cat([x,f],1)
#         x=self.conv(x)
#         x=self.cwr(x)*self.swr(x)
#         x=x+f

#         return x


class sscrDetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.sscr = nn.ModuleList(SSCR(x) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        f = x[self.nl:]
        x= x[:self.nl]
        for i in range(self.nl):
            x[i] = self.m[i](self.sscr[i](x[i],f[i]))  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
