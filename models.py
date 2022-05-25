import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision
import torchvision.models as models


class Attention_Encoder(nn.Module):
    def __init__(self, encode_size):
        """
        Args:
            encoded_size: int
              size(height and width) of encoded image (e.g. 14)
        """
        super(Attention_Encoder, self).__init__()

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
    def __init__(self, encoder_outdim: int, decoder_indim: int, decoder_outdim: int):
        """
        Args:

        """
        super(SoftAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_outdim, decoder_indim)
        self.decoder_att = nn.Linear(decoder_outdim, decoder_indim)
        self.total_att = nn.Linear(decoder_indim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, feature: Tensor, hidden_prev: Tensor):
        """
        Args:
            feature: torch.Tensor((batch_size, encoder_channel, encode_size, encode_size))
                features extracted from CNN, resized
            hidden_prev: torch.Tensor((batch_size, decoder_outdim))
                hidden vector extracted from RNN previous step
        """
        batch_size, encode_channel = feature.size(0), feature.size(1)
        feature = feature.permute(0, 2, 3, 1).view(
            batch_size, -1, encode_channel
        )  # (batch_size, n_pixels, encode_channel)
        att1 = self.encoder_att(feature)  # (batch_size, n_pixels, decoder_indim)
        att2 = self.decoder_att(hidden_prev)  # (batch_size, decode_indim)
        att = self.total_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, n_pixels)
        alpha = self.softmax(att)  # (batch_size, n_pixels)
        attention_weighted_encoding = (feature * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encode_channel)

        return attention_weighted_encoding, alpha


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
