import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hidden_dim, n):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n = n

        res_layers = 3
        layers = 3
        cur_dim = hidden_dim

        self.layers = [nn.Conv2d(3, hidden_dim, 3,1,1)]

        for i in range(layers):
            input_dim = cur_dim
            output_dim = input_dim * 2
            for j in range(res_layers):
                self.layers.append(Residual(input_dim, output_dim))
                input_dim = output_dim
            self.layers.append(nn.Conv2d(input_dim, output_dim, 4,2,1))
            cur_dim = output_dim

        self.layers.append(nn.Conv2d(cur_dim, n, 3,1,1))
        self.model = nn.Sequential(*self.layers)

        # self.cnn1 = nn.Conv2d(3, hidden_dim, kernel_size = 4, stride = 2)
        # self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim*2,kernel_size = 4, stride = 2)
        # self.cnn3 = nn.Conv2d(hidden_dim*2, n, kernel_size = 4, stride = 2)
        #
        # self.res1 = Residual(hidden_dim*2)
        # self.res2 = Residual(hidden_dim*2)
        #
        # self.relu = nn.ReLU()


    def forward(self,x):
        x = self.model(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        if in_channels != out_channels:

            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self,x):
        res = x

        x = self.cnn1(self.relu(x))
        x = self.cnn2(self.relu(x))

        if self.in_channels != self.out_channels:
            res = self.channel_up(res)

        return x + res


# encoder = Encoder(32, 128).to('cuda')
# print(sum(p.numel() for p in encoder.parameters()))
# dummy_input = torch.randn(1, 3, 128, 128).to('cuda')
# latent_output = encoder(dummy_input)
#
# print(latent_output.shape)
#
