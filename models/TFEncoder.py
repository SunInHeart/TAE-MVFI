import torch
import torch.nn as nn
from models.TFLayer import TFLayer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


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


class TFEncoder(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 embed_dim=[32, 64, 128, 256], depths=[2, 6, 6, 6], num_heads=[2, 4, 8, 16],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, resi_connection='1conv', use_crossattn=None):
        super(TFEncoder, self).__init__()

        num_in_ch = in_chans
        num_out_ch = out_chans

        self.stem = nn.Sequential(
            nn.Conv2d(num_in_ch, embed_dim[0], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2),
            ResBlock(embed_dim[0], kernel_size=3),
        )

        self.down0 = nn.Sequential(nn.Conv2d(embed_dim[0], embed_dim[0], 3, 2, 1),
                                   # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   # nn.Conv2d(embed_dim[0], embed_dim[0], 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True), )
        self.down1 = nn.Sequential(nn.Conv2d(embed_dim[0], embed_dim[1], 3, 2, 1),
                                   # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   # nn.Conv2d(embed_dim[1], embed_dim[1], 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True), )
        self.down2 = nn.Sequential(nn.Conv2d(embed_dim[1], embed_dim[2], 3, 2, 1),
                                   # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   # nn.Conv2d(embed_dim[2], embed_dim[2], 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True), )
        self.down3 = nn.Sequential(nn.Conv2d(embed_dim[2], embed_dim[3], 3, 2, 1),
                                   # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   # nn.Conv2d(embed_dim[3], embed_dim[3], 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True), )

        # TFBs = []
        TFBs = nn.ModuleList()
        for i in range(len(depths)):
            TFBs.append(TFB(dim=embed_dim[i],  # 32
                            depth=depths[i],  # 2
                            num_heads=num_heads[i],  # 2
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            norm_layer=norm_layer,
                            resi_connection=resi_connection,
                            use_crossattn=use_crossattn[i]  # [True, True, True, True, True, True, True, True]
                            )
                        )

        self.stage1 = TFBs[0]
        self.stage2 = TFBs[1]
        self.stage3 = TFBs[2]
        self.stage4 = TFBs[3]


    def forward(self, x):
        x0 = self.stem(x.contiguous())  # 1

        x1 = self.down0(x0)  # 1->1/2
        x1 = self.stage1(x1)
        # x1 = self.forward_features(x1, self.layers0)

        x2 = self.down1(x1)  # 1/2->1/4
        x2 = self.stage2(x2)
        # x2 = self.forward_features(x2, self.layers1)

        x3 = self.down2(x2)  # 1/4->1/8
        x3 = self.stage3(x3)
        # x3 = self.forward_features(x3, self.layers2)

        x4 = self.down3(x3)  # 1/8->1/16
        x4 = self.stage4(x4)
        # x4 = self.forward_features(x4, self.layers3)

        return x1, x2, x3, x4


class TFB(nn.Module):
    def __init__(self, dim, depth=2, num_heads=2, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, resi_connection='1conv', use_crossattn=None):
        super(TFB, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = TFLayer(dim=dim,
                            num_heads=num_heads, window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            norm_layer=norm_layer,
                            use_crossattn=use_crossattn[i])  # False
            self.layers.append(layer)

    def forward(self, x):
        # for block in self.blocks:
        #     x = block(x) + x

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')

        for layer in self.layers:
            x = layer(x)
        x = x.view(B, H, W, -1)

        # x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x
