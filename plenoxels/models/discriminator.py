import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


def reparametrize(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x, dim=-1):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, dim)


class Discriminator(nn.Module):
    def __init__(self, conditional, policy, ndf=64, imsize=64,nc=3):
        super(Discriminator, self).__init__()
        # assert (imsize == 16 or imsize == 32 or imsize == 64 or imsize == 128)

        nc = nc
        self.conditional = conditional
        self.policy = policy
        self.imsize = imsize

        SN = torch.nn.utils.spectral_norm
        def IN(x): return nn.InstanceNorm2d(x)

        if self.conditional:
            final_dim = ndf
            self.embedding_scale = Embedding(1, 4)

            self.final = nn.Sequential(
                nn.LeakyReLU(0.2),
                SN(nn.Conv2d(ndf + self.embedding_scale.out_channels,
                   ndf, (1, 1), (1, 1), (0, 0), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                SN(nn.Conv2d(ndf, ndf, (1, 1), (1, 1), (0, 0), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                SN(nn.Conv2d(ndf, 1, (1, 1), (1, 1), (0, 0), bias=False)),
            )
        else:
            final_dim = 1

        blocks = []
        if self.imsize == 128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(nc, ndf // 2, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf // 2, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(nc, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 32:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(nc, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(nc, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            # nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(ndf * 8, final_dim, (4, 4), (1, 1), (0, 0), bias=False)),
            # nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

    def forward(self, input, y=None):
        if self.policy is not None and np.random.random() > 0.5:
            input = DiffAugment(input, policy=self.policy)
        else:
            input = input.contiguous()

        input = self.main(input)  # [N, c1, 1, 1]

        if self.conditional:
            y = self.embedding_scale(y, dim=1)
            input = torch.cat((input, y), 1)  # [N, c1+c2, 1, 1]
            input = self.final(input).flatten()

        return input


# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738


# def DiffAugment(x, policy = 'translation,cutout', channels_first=True):
# def DiffAugment(x, policy = 'color,translation,cutout', channels_first=True):


def DiffAugment(x, policy='color,cutout', channels_first=True):
    if np.random.random() < 0.5:
        return x
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                   dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                   dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio +
                           0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1,
                                  size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1,
                                  size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[
        grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(
        2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(
        3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x -
                         cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y -
                         cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3),
                      dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
