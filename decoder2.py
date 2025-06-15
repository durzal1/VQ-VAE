import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish
import torch

class Decoder2(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder2, self).__init__()
        channels = [256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, stride=1,padding=0))
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, stride=1,padding=0))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, 3, 3, 1, 1))
        # layers.append(nn.Conv2d(3, 3, 1, 1, 2))
        # layers.append(nn.Conv2d(3, 3, 1, 1, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder3(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder3, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0 and i!= 1:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, 3, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# encoder = Decoder3(64).to('cuda')
# dummy_input = torch.randn(1, 64, 16, 16).to('cuda')
# latent_output = encoder(dummy_input)
# print(latent_output.shape)