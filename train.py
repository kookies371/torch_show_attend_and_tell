from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from models import TotalModel
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
    model: TotalModel,
    dataset: CustomCOCOCaption,
    loss_fn,
    optimizer: torch.optim.Optimizer,
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
    D = model.D
    L = model.L # n_pixels after encode
    # max_cpi: maximum number of captions per image
    # max_cap_len: maximum number of words in each caption
    # num_words: number of words appeared in the dataset = size of embedding vector
    max_cpi = dataset.max_cpi
    max_cap_len = dataset.max_cap_len # C
    num_words = dataset.num_words # K

    dataloader = DataLoader(
        dataset,
        batch_size=N,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(max_epoch):
        for i_batch, (imgs, captionss) in enumerate(dataloader):
            # imgs: Tensor(N, 3, 640, 640)
            # captionss: LongTensor(N, C) (index tensor, not one-hot)
            imgs = imgs.to(device)
            captionss = captionss.to(device)

            # initialize optimizer
            optimizer.zero_grad()

            # forward img and get feature map
            # feature: Tensor(N, L, D)
            feature = model.encoder(imgs) # (N, D, encode_w, encode_h)
            feature = feature.view(N, D, -1) # (N, D, L)
            feature = feature.permute(0, 2, 1) # (N, L, D)

            # Initialize hidden vectors
            # prev_h, prev_c: Tensor(N, n)
            prev_h, prev_c = model.init_hiddens(feature)

            # initialize Tensor storing results
            # predictions: Tensor(N, C, K)
            # alphas: Tensor(N, L, C)
            predictions = torch.zeros((N, max_cap_len, num_words), dtype=torch.float).to(device)
            alphas = torch.zeros((N, L, max_cap_len)).to(device)

            # forward feature map in LSTM
            # variable name 't' from paper
            for t in range(max_cap_len):
                # calculate weighted feature through attention layer
                # context: Tensor(N, D) (=weighted_feature, 'z_hat' in paper)
                # alpha: Tensor(N, L)
                context, alpha = model.attention(feature, prev_h)
                
                # prepare y
                # prev_y: Tensor(N, K)
                # prev_y_embed: Tensor(N, m)
                prev_y = F.one_hot(captionss[:, t], num_classes=num_words)
                prev_y = prev_y.type(torch.float)
                prev_y_embed = model.E(prev_y)

                # decode through LSTM cell
                next_h, next_c = model.decoder(
                    torch.cat([context, prev_y_embed], dim=1),
                    (prev_h, prev_c)
                )

                # calculate output word probability
                # p: Tensor(N, m)
                p = model.E(prev_y) + model.Lh(next_h) + model.Lz(context)
                p = model.Lo(p)
                p = F.softmax(p, dim=1)

                # stack results
                predictions[:, t, :] = p
                alphas[:, :, t] = alpha

                # update hidden vectors
                prev_h, prev_c = next_h, next_c

            # train
            loss = loss_fn(predictions.permute(0, 2, 1), captionss)
            loss.backward()
            optimizer.step()

            # make statistics?
            print(f'epoch: {epoch}, i_batch: {i_batch}, loss: {loss.item()}')


if __name__ == "__main__":
    # arg parser
    # batch_size, max_epoch, loss, optimizer 등 학습 정보 입력받기 (default도)
    # max_cpi, max_cap_len, attention type 등 모델 정보
    # model을 저장할 filename도

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model, dataset
    model = TotalModel(D=2048, K=10000, C=100, L=196, m=100, n=100).to(device)
    dataset = CustomCOCOCaption(
        root='../torch_datasets/val2014',
        annFile='../torch_datasets/annotations/captions_val2014.json',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640))
        ])
    )

    # optimizer, loss_fn
    max_epoch = 100
    batch_size = 4
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
    train(model, dataset, loss_fn, optimizer, batch_size, max_epoch)

    # save model
    save_model = False
    if save_model:
        pass
