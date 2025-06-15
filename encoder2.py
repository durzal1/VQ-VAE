import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish
import torch

class Encoder2(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2, self).__init__()
        channels = [64, 64, 128, 128]
        num_res_blocks = 1
        resolution = 256
        layers = [nn.Conv2d(3, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

            layers.append(DownSampleBlock(channels[i+1]))
            resolution //= 2

        layers.append(nn.Conv2d(channels[-1], latent_dim, 3, 1, 0))
        layers.append(nn.Conv2d(latent_dim, latent_dim, 4, 1, 0))
        layers.append(nn.Conv2d(latent_dim, latent_dim, 1, 1, 1))
        layers.append(nn.Conv2d(latent_dim, latent_dim, 1, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class Encoder3(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder3, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(3, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2 and i != len(channels) - 3:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# encoder = Encoder3( 64).to('cuda')
# print(sum(p.numel() for p in encoder.parameters()))
# dummy_input = torch.randn(1, 3, 128, 128).to('cuda')
# latent_output = encoder(dummy_input)
#
# print(latent_output.shape)
