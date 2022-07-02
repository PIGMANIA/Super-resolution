import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, bias=True, padding=3 // 2)
            )
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res * self.res_scale


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        scale = cfg.scale
        num_in_ch = cfg.num_in_ch
        num_out_ch = cfg.num_out_ch
        num_feat = cfg.num_feat
        num_block = cfg.num_block
        res_scale = cfg.res_scale

        self.head = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=3 // 2)
        body = [ResBlock(num_feat, res_scale) for _ in range(num_block)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(
                num_feat,
                num_feat * (scale**2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(scale),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
