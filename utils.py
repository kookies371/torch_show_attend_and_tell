import torchvision.datasets as datasets

from tqdm import tqdm
from collections import Counter


def count_words(dataset: datasets, tokenizer) -> Counter:
    word_counter = Counter()

    for i, (_, captions) in enumerate(tqdm(dataset)):
        for caption in captions:
            word_counter += Counter(tokenizer(caption))
        
    return word_counter