import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.transformer_layers import TFModel
from models import dsepconv

class Network(nn.Module):
    def __init__(self, Generated_ks=5, useBias=True, isMultiple=False):
        super(Network, self).__init__()
        window_size = 8
        embed_dim = [32, 64, 128, 256]
        # embed_dim = [128, 192, 256, 320]
        # embed_dim = [64, 128, 256, 512]

        self.isMultiple = isMultiple

        self.transformer = TFModel(in_chans=6, out_chans=64,
                                   window_size=window_size, img_range=1.,
                                   # depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                   depths=[2, 6, 6, 6],
                                   embed_dim=embed_dim, num_heads=[2, 4, 8, 16], mlp_ratio=2,
                                   resi_connection='1conv',
                                   use_crossattn=[[True, True, True, True, True, True, True, True],
                                                  [True, True, True, True, True, True, True, True],
                                                  [True, True, True, True, True, True, True, True],
                                                  [True, True, True, True, True, True, True, True]])

        self.apply(self._init_weights)

        self.predict_ll = SynBlock(Generated_ks=Generated_ks, useBias=useBias, isMultiple=isMultiple)
        self.predict_l = SynBlock(Generated_ks=Generated_ks, useBias=useBias, isMultiple=isMultiple)
        self.predict = SynBlock(Generated_ks=Generated_ks, useBias=useBias, isMultiple=isMultiple)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, tensors):
        img0 = tensors[0]
        img1 = tensors[1]

        B, _, H, W = img0.size()
        x = torch.cat((img0, img1), dim=1)

        o_1, o_2, o_3 = self.transformer(x)
        tensorCombine3 = o_3
        tensorCombine2 = o_2
        tensorCombine1 = o_1

        tensors_ll = []
        tensors_l = []

        for i in range(len(tensors)):
            tensors_ll.append(F.interpolate(tensors[i], scale_factor=1 / 4, mode='bilinear'))
            tensors_l.append(F.interpolate(tensors[i], scale_factor=1 / 2, mode='bilinear'))

        out_ll = self.predict_ll(tensorCombine3, tensors_ll)

        out_l = self.predict_l(tensorCombine2, tensors_l)
        out_l = F.interpolate(out_ll, size=out_l.size()[-2:], mode='bilinear') + out_l

        out = self.predict(tensorCombine1, tensors)
        out = F.interpolate(out_l, out.size()[-2:], mode='bilinear') + out

        return out


class SynBlock(nn.Module):
    def __init__(self, Generated_ks, useBias, isMultiple):
        super(SynBlock, self).__init__()
        self.generated_ks = Generated_ks
        self.useBias = useBias
        self.isMultiple = isMultiple
        if self.isMultiple:
            self.estimator_in = 65
        else:
            self.estimator_in = 64


        def KernelNet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=self.generated_ks, out_channels=self.generated_ks, kernel_size=3, stride=1,
                                padding=1)
            )

        # end

        def Offsetnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1,
                                padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3,
                                stride=1, padding=1)
            )

        def Masknet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1,
                                padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3,
                                stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        def Biasnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            )

        self.moduleVertical1 = KernelNet()
        self.moduleVertical2 = KernelNet()
        self.moduleHorizontal1 = KernelNet()
        self.moduleHorizontal2 = KernelNet()

        self.moduleOffset1x = Offsetnet()
        self.moduleOffset1y = Offsetnet()
        self.moduleOffset2x = Offsetnet()
        self.moduleOffset2y = Offsetnet()

        self.moduleMask1 = Masknet()
        self.moduleMask2 = Masknet()

        self.moduleBias = Biasnet()

    def forward(self, tensorCombine, tensors):
        tensorFirst = tensors[0]
        tensorSecond = tensors[1]
        tensorFirst = torch.nn.functional.pad(input=tensorFirst,
                                              pad=[int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0)),
                                                   int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0))],
                                              mode='replicate')
        tensorSecond = torch.nn.functional.pad(input=tensorSecond,
                                               pad=[int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0)),
                                                    int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0))],
                                               mode='replicate')
        if self.isMultiple:
            tensorTime = tensors[2]
            v1 = self.moduleVertical1(torch.cat([tensorCombine, tensorTime], 1))
            v2 = self.moduleVertical2(torch.cat([tensorCombine, 1. - tensorTime], 1))
            h1 = self.moduleHorizontal1(torch.cat([tensorCombine, tensorTime], 1))
            h2 = self.moduleHorizontal2(torch.cat([tensorCombine, 1. - tensorTime], 1))
            offset1x = self.moduleOffset1x(torch.cat([tensorCombine, tensorTime], 1))
            offset1y = self.moduleOffset1y(torch.cat([tensorCombine, tensorTime], 1))
            offset2x = self.moduleOffset2x(torch.cat([tensorCombine, 1. - tensorTime], 1))
            offset2y = self.moduleOffset2y(torch.cat([tensorCombine, 1. - tensorTime], 1))
            mask1 = self.moduleMask1(torch.cat([tensorCombine, tensorTime], 1))
            mask2 = self.moduleMask2(torch.cat([tensorCombine, 1. - tensorTime], 1))

        else:
            v1 = self.moduleVertical1(tensorCombine)
            v2 = self.moduleVertical2(tensorCombine)
            h1 = self.moduleHorizontal1(tensorCombine)
            h2 = self.moduleHorizontal2(tensorCombine)
            offset1x = self.moduleOffset1x(tensorCombine)
            offset1y = self.moduleOffset1y(tensorCombine)
            offset2x = self.moduleOffset2x(tensorCombine)
            offset2y = self.moduleOffset2y(tensorCombine)
            mask1 = self.moduleMask1(tensorCombine)
            mask2 = self.moduleMask2(tensorCombine)

        tensorDot1 = dsepconv.FunctionDSepconv(tensorFirst, v1, h1, offset1x, offset1y, mask1)
        tensorDot2 = dsepconv.FunctionDSepconv(tensorSecond, v2, h2, offset2x, offset2y, mask2)

        # if self.useBias:
        #     return tensorDot1 + tensorDot2 + self.moduleBias(tensorCombine)
        # else:
        #     return tensorDot1 + tensorDot2

        return tensorDot1 + tensorDot2



