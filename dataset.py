import torch
import torch.nn.functional as F
from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
from torchtext.data import get_tokenizer
import json

class CustomCOCOCaption(CocoCaptions):
    """CustomCOCOCaption Dataset"""

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: transforms = None,
        max_cpi: int = 10,
        max_cap_len: int = 100,
        num_words: int = 10000,
        tokenizer: str = "basic_english",
    ):
        """
        Args:
            root, annFile: pytorch official tutorial
            max_cpi: int
                maximum number of captions per image.
                max_cpi of COCOCaption dataset is 7.
            max_cap_len: int
                maximum number of words in each caption
                max_cap_len of COCOCaption dataset is 57.
            num_words: int
                number of words appeared in the dataset
                num_words of COCOCaption dataset is 7581 include <start>, <pad>, <unk>, <end>.
        """
        super(CustomCOCOCaption, self).__init__(root, annFile, transform=transform)
        
        self.max_cpi = max_cpi
        self.max_cap_len = max_cap_len
        self.num_words = num_words
        self.tokenizer = get_tokenizer(tokenizer)

        with open("./word_map.json") as f:
            self.word_map = json.load(f)


    def _collate_fn(self, x):
        imgs, captionss = [], []
        for img, captions in x:
            imgs.append(img)

            tokens = self.tokenizer(captions)
            tokens = [self.word_map[token] for token in tokens]
            tokens = [self.word_map["<start>"]] + tokens + [self.word_map["<end>"]]
            tokens = tokens + [self.word_map["<pad>"]] * (self.max_cap_len - len(tokens))
            captionss.append(tokens)
        
        imgs = torch.stack(imgs, dim=0)
        captionss = torch.stack(tokens, dim=0)

        return (imgs, captionss)