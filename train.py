import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset import CustomCOCOCaption

"""TODO
모델에 들어가야 디폴트 정보들
max_caption_length
batch_size
attention type

데이터셋에 들어가야 할 정보들
max_cpi
word_map

utils
datasete collate_fn
make_idx

"""


def train(
    model,
    dataset: CustomCOCOCaption,
    loss_fn,
    optimizer,
    batch_size: int = 16,
    max_epoch: int = 100,
) -> None:
    """Train attention model

    Args:
        model: Attention Model with Encoder, Decoder, Attention
        dataset: Custom Dataset with (imgs, captionss)
            imgs: Tensor(N, C, W, H)
            captionss: N * captions
                captions: (true_caption1, true_caption2, ...)
        max_epoch: int
            maximum train epoch
        loss_fn: torch.nn.losses
        optimizer: torch.optim
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    N = batch_size
    # max_cpi: maximum number of captions per image
    # max_cap_len: maximum number of words in each caption
    # num_words: number of words appeared in the dataset = size of embedding vector
    max_cpi = dataset.max_cpi
    max_cap_len = dataset.max_cap_len
    num_words = dataset.num_words

    dataloader = DataLoader(
        dataset,
        batch_size=N,
        collate_fn=dataset._collate_fn(),
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(max_epoch):
        for i_batch, (imgs, captionss) in enumerate(dataloader):
            # imgs : Tensor(N, 3, 640, 640)
            # captionss : Tensor(N, max_cpi, max_cap_len)

            # forward img and get feature map
            # may feature_h = feature_w
            imgs = imgs.to(device)
            features = model.Encoder(imgs)  # (N, feature_c, feature_h, feature_w)

            # Initialize hidden vectors
            # h, c: Tensor(N, decoder_hidden)
            prev_h, prev_c = model.init_hidden(features)
            # y: Tensor(N, )
            prev_y = dataset.word_map["<start>"]

            # forward feature map in LSTM
            # variable name 't' from paper
            for t in range(max_cap_len):
                # calculate weighted feature through attention layer
                # decoder_in = feature_c
                # n_pixels = feature_h * feature_w
                alpha = model.Attention(features, h)  # (N, n_pixels)
                context = model.calculate_context(features, alpha)  # (N, decoder_in)

                # decode through LSTM cell
                h, c = model.Decoder(context, (h, c))

                # calculate output word probability
                if t == 0:
                    prev_y = word_map["<start>"]  # TODO : shape?
                else:
                    prev_y = idx[..., t - 1]
                probability = model.embedding(prev_y) + model.Lh(h) + model.Lz(context)
                probability = model.L_o(probability).softmax(dim=1)  # (N, embed_size)

                # stack results
                predictions[..., t] = probability

            # calculate loss
            loss = loss_fn(predictions)

            # backpropagate loss

            # make statistics?


if __name__ == "__main__":
    # arg parser
    # batch_size, max_epoch, loss, optimizer 등 학습 정보 입력받기 (default도)
    # max_cpi, max_cap_len, attention type 등 모델 정보
    # model을 저장할 filename도

    # Initialize model, dataset
    model = 0
    dataset = CustomCOCOCaption()

    # optimizer, loss_fn
    max_epoch = 100
    batch_size = 16
    loss_fn = 0
    optimizer = 0
    train(model, dataset, loss_fn, optimizer, batch_size, max_epoch)

    # save model
    save_model = False
    if save_model:
        pass
