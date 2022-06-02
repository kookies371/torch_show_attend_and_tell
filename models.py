from functools import total_ordering
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision
import torchvision.models as models


class AttentionEncoder(nn.Module):
    def __init__(self, encode_size):
        """
        Args:
            encoded_size: int
              size(height and width) of encoded image (e.g. 14)
        """
        super(AttentionEncoder, self).__init__()

        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_size, encode_size))

    def forward(self, imgs: Tensor) -> Tensor:
        """CNN Encoder
        Args:
            imgs: Tensor(batch_size, 3, W, H)

        Return:
            out: Tensor(N, D, encode_size, encode_size)
                features extraced from CNN
                L = encode_size * encode_size
        """
        out = self.resnet(imgs)
        out = self.adaptive_pool(out)
        return out


class SoftAttention(nn.Module):
    def __init__(self, D: int, n: int, attention_dim: int):
        super(SoftAttention, self).__init__()
        self.encoder_att = nn.Linear(D, attention_dim)
        self.decoder_att = nn.Linear(n, attention_dim)
        self.total_att = nn.Linear(attention_dim, 1)

    def forward(self, feature: Tensor, prev_h: Tensor):
        """
        Args:
            feature: torch.Tensor((N, L, D))
                features extracted from CNN, resized
            prev_h: torch.Tensor((N, n))
                hidden vector extracted from RNN previous step
        """
        att1 = self.encoder_att(feature)  # (N, L, attention_dim)
        att2 = self.decoder_att(prev_h)  # (N, attention_dim)

        att = att1 + torch.unsqueeze(att2, dim=1) # (N, L, attention_dim)
        att = F.relu(att) # (N, L, attention_dim)
        att = self.total_att(att) # (N, L, 1)
        att = torch.squeeze(att, 2) # (N, L)

        alpha = F.softmax(att, dim=1) # (N, L)

        weighted_feature = feature * alpha.unsqueeze(2) # (N, L, D)
        weighted_feature = weighted_feature.sum(dim=1) # (N, D)

        return weighted_feature, alpha


class HardAttention(nn.Module):
    def __init__(self):
        super(HardAttention, self).__init__()
        pass

    def forward(self):
        pass


class AttentionDecoder(nn.Module):
    """LSTM Decoder"""
    def __init__(self, D: int, m: int, n: int):
        """
        Args:
            D: size of context vector
            m: embedding dimension
            n: LSTM hidden dimension
        """
        super(AttentionDecoder, self).__init__()

        self.LSTM = nn.LSTMCell(D + m, n)

    def forward(self, input, hiddens):
        hidden_h, hidden_c = hiddens
        return self.LSTM(input, (hidden_h, hidden_c))
        

class TotalModel(nn.Module):
    def __init__(self, D, K, C, L, m, n, att_type='soft', encoder_size=14):
        """
        Args:
            D: 
                size of context vector
                channel of encoded image
            K:
                size of vocabulary
                size of word embedding vector
            C:
                max length of the caption
            L:
                n_pixels after encoding
            m:
                embedding dimensionality
            n:
                LSTM dimensionality
                size of hidden vector of LSTM
        
        Raises:
            ValueError: attention module
        """
        super(TotalModel, self).__init__()

        # constants
        self.D = D
        self.K = K
        self.C = C
        self.L = L
        self.m = m
        self.n = n

        # embedding layer
        self.E = nn.Linear(K, m)

        # encoder
        self.encoder = AttentionEncoder(encoder_size)

        # attention module
        if att_type.lower() == 'soft':
            self.attention = SoftAttention(D, n, 100)
        elif att_type.lower() == 'hard':
            self.attention = HardAttention(D, n, 100)
        else:
            raise ValueError(f'type of attention module must be "soft" or "hard", not {att_type}')
        
        # MLP that initialize prev_c and prev_h
        self.init_h = nn.Linear(D, n)
        self.init_c = nn.Linear(D, n)

        # decoder
        self.decoder = AttentionDecoder(D, m, n)

        # calculate probability layers
        self.Lo = nn.Linear(m, K)
        self.Lh = nn.Linear(n, m)
        self.Lz = nn.Linear(D, m)

    
    def init_hiddens(self, feature: Tensor):
        """
        Args:
            feature: Tensor(N, L, D)
        Returs:
            c0, h0: Tensor(N, n)
        """
        feature_mean = feature.mean(dim=1) # (N, D)
        h0 = self.init_h(feature_mean)
        c0 = self.init_c(feature_mean)
        return h0, c0
