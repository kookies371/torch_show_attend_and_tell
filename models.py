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

        # Load pretrained resnet and drop two layers
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Fit size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_size, encode_size))

        # Fine tune?

    def forward(self, imgs: Tensor) -> Tensor:
        """CNN Encoder
        Args:
            imgs: torch.Tensor((batch_size, 3, W, H))

        Return:
            out: torch.Tensor((batch_size, encode_channel, encode_size, encode_size))
                features extraced from CNN
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

        att = att1 + att2.unsqueeze(1) # (N, L, attention_dim)
        att = att.relu() # (N, L, attention_dim)
        att = self.total_att(att) # (N, L, 1)
        att = att.squeeze(2) # (N, L)

        alpha = self.softmax(att) # (N, L)

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

    def __init__(
        self, input_dim, output_dim, attention_module, vocab_size, embedding_size=2048
    ):
        """
        Args:
            input_dim:
            output_dim:
            attention_module:
            vocab_size:
        """
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size

        self.init_memory_MLP = nn.Linear(input_dim, output_dim)
        self.init_hidden_MLP = nn.Linear(input_dim, output_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.attention_module = attention_module

        self.LSTM = nn.LSTM(output_dim, batch_first=True)
        pass

    def forward(self, feature: Tensor):
        """
        Args:
            feature: torch.Tensor((batch_size, encode_channel, encode_size, encode_size))
        """
        # prerequisites
        batch_size = feature.size(0)
        vocab_size = self.vocab_size

        meaned_feature = torch.mean(feature, dim=1).view(
            batch_size, -1
        )  # (batch_size, encode_size * encode_size)
        initial_memory = self.init_memory_MLP(
            meaned_feature
        )  # (batch_size, decoder_dim)
        initial_hidden = self.init_hidden_MLP(meaned_feature)

        context = self.attention_module(feature, initial_hidden)

        predictions = torch.zeros((batch_size, max_length, vocab_size))

        pass

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
            self.attention = SoftAttention()
        elif att_type.lower() == 'hard':
            self.attention = HardAttention()
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


    def forward(self, images: Tensor):
        batch_size = images.shape[0]

        # calculate feature vectore (vector 'a' in paper)
        # feature: Tensor(N, L, D)
        feature = self.encoder(images)
        feature = feature.view(batch_size, self.D, -1)
        feature = feature.permute(0, 2, 1)

        # initialize previous informations
        # prev_h, prev_c: Tensor(N, n) ('h0', 'c0' in paper)
        # prev_word: Tensor(N, K) ('y_{t-1}' in paper, one-hot)
        prev_h, prev_c = self._init_hiddens(feature)
        prev_word = torch.zeros((batch_size, self.K)) # TODO!!

        # run LSTM decoder
        for word_i in range(self.C):
            # calculate attention
            # context: torch.Tensor((N, D)) (=weighted_feature)
            # alpha: torch.Tensor((N, L))
            context, alpha = self.attention(feature, prev_h)

            # decode
            # decoder_input: Tensor(N, D + m)
            decoder_input = torch.cat([context, self.E(prev_word)], dim=1)
            next_h, next_c = self.decoder(decoder_input, (prev_h, prev_c))

            # update hidden vectors
            prev_h, prev_c = next_h, next_c

            # TODO: update prev_word

    
    def _init_hiddens(self, feature: Tensor):
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