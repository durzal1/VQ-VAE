import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from decoder2 import Decoder2
from Codebook import Codebook
from encoder2 import Encoder2

class VQModel(nn.Module):
    def __init__(self, n, k, hidden_dim, beta):
        super(VQModel, self).__init__()
        embedding_dim = n
        self.beta = beta

        self.encoder = Encoder(hidden_dim, embedding_dim)
        # self.encoder = Encoder2(embedding_dim)
        self.codebook = Codebook(n, k, beta)
        self.decoder = Decoder(hidden_dim, embedding_dim)
        # self.decoder = Decoder2(embedding_dim)


    def forward(self,x):
        '''

        :param x: shape (batch, 3, h, w)
        :return: shape (batch,  3 , h, w)
        '''

        x = self.encoder(x)
        x, loss, min_encoding_indices = self.codebook(x)
        x = self.decoder(x)

        return x,loss, min_encoding_indices

