# -----------------------------------------------------------------------------------
# modified from: 
# SwinIR: Image Restoration Using Swin Transformer, https://github.com/JingyunLiang/SwinIR
# -----------------------------------------------------------------------------------
import sys
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
sys.path.append('../..')


class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        x += res
        return self.relu(x)


class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.upconv = nn.ModuleList(
                [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                 ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x


class TFModel(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 embed_dim=[32, 64, 128, 256], depths=[2, 6, 6, 6], num_heads=[2, 4, 8, 16],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 resi_connection='1conv', use_crossattn=None,
                 **kwargs):
        super(TFModel, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        from models.TFEncoder import TFEncoder
        self.encoder = TFEncoder(in_chans=in_chans, out_chans=out_chans,
                                 embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer,
                                 resi_connection=resi_connection, use_crossattn=use_crossattn)

        self.decoder = nn.Sequential(
            upSplit(embed_dim[-1], embed_dim[-2]),
            upSplit(embed_dim[-2] * 2, embed_dim[-3]),
            upSplit(embed_dim[-3] * 2, embed_dim[-4]),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self.conv_up0 = nn.Sequential(nn.ConvTranspose2d(embed_dim[-1], embed_dim[-2], 4, 2, 1),
        #                               nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.conv_up1 = nn.Sequential(nn.ConvTranspose2d(2 * embed_dim[-2], embed_dim[-3], 4, 2, 1),
        #                               nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.conv_up2 = nn.Sequential(nn.ConvTranspose2d(2 * embed_dim[-3], embed_dim[-4], 4, 2, 1),
        #                               nn.LeakyReLU(negative_slope=0.2, inplace=True))

        def SmoothNet(inc, outc):
            return torch.nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
                ResBlock(outc, kernel_size=3),
            )

        self.smooth0 = SmoothNet(2*embed_dim[-2], num_out_ch)
        self.smooth1 = SmoothNet(2*embed_dim[-3], num_out_ch)
        self.smooth2 = SmoothNet(2*embed_dim[-4], num_out_ch)

        # self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward(self, x):

        x1, x2, x3, x4 = self.encoder(x)

        # dx3 = self.conv_up0(x4)  # 1/16->1/8
        dx3 = self.lrelu(self.decoder[0](x4, x3.size()))
        dx3 = torch.cat([dx3, x3], dim=1)

        # dx2 = self.conv_up1(dx3)  # 1/8->1/4
        dx2 = self.lrelu(self.decoder[1](dx3, x2.size()))
        dx2 = torch.cat([dx2, x2], dim=1)

        # dx1 = self.conv_up2(dx2)  # 1/4->1/2
        dx1 = self.lrelu(self.decoder[2](dx2, x1.size()))
        dx1 = torch.cat([dx1, x1], dim=1)

        o3 = self.smooth0(dx3)
        o2 = self.smooth1(dx2)
        o1 = self.smooth2(dx1)

        return o1, o2, o3

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops




if __name__ == '__main__':
    device = 'cuda'
    window_size = 8
    height = 128
    width = 128
    model = SwinIR(img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2).to(device)
    # print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width)).to(device)
    # x0 = torch.randn((1, 3, height, width)).to(device)
    # x1 = torch.randn((1, 3, height, width)).to(device)
    x = model(x)
    print(x.shape)
