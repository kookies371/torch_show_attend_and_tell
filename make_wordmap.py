import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchtext.data import get_tokenizer

import json

from utils import count_words


def make_wordmap(word_count_thr=10):
    # prepare datasets
    train_dataset = datasets.CocoCaptions(
        root="../torch_datasets/train2014",
        annFile="../torch_datasets/annotations/captions_train2014.json",
        transform=transforms.ToTensor(),
    )
    val_dataset = datasets.CocoCaptions(
        root="../torch_datasets/val2014",
        annFile="../torch_datasets/annotations/captions_val2014.json",
        transform=transforms.ToTensor(),
    )

    # prepare tokenizer
    tokenizer = get_tokenizer("basic_english")

    # count
    train_counter = count_words(train_dataset, tokenizer)
    val_counter = count_words(val_dataset, tokenizer)
    total_counter = train_counter + val_counter

    # counter to wordmap
    word_map = dict()
    word_map["<start>"] = 0
    word_map["<end>"] = 1
    word_map["<pad>"] = 2
    word_map["<unk>"] = 3

    idx = 4
    for word, count in total_counter.items():
        if not count < word_count_thr:
            word_map[word] = idx
            idx += 1

    return word_map


if __name__ == "__main__":
    word_map = make_wordmap()

    # save word_map
    with open("word_map.json", "w") as f:
        json.dump(word_map, f)