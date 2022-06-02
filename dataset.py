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
        max_cap_len: int = 100, # C
        num_words: int = 10000, # K
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


    def collate_fn(self, x):
        imgs, captionss = [], []
        for img, captions in x:
            # img: Tensor(3, 640, 640)
            # captions: ['caption1', 'caption2', ...]
            imgs.append(img)

            for caption in captions:
                # caption: 'sentence'
                # idxs: [int] size C(=max_cap_len)
                #   [0, 3, 172, 15, ...] corresponding to word_map
                tokens = self.tokenizer(caption)

                idxs = [self.word_map.get(token, self.word_map["<unk>"]) for token in tokens]
                idxs = [self.word_map["<start>"]] + idxs + [self.word_map["<end>"]]
                idxs = idxs + [self.word_map["<pad>"]] * (self.max_cap_len - len(idxs))

            captionss.append(idxs)
        
        # imgs: Tensor(N, 3, 640, 640)
        # idxss: Tensor(N, C)
        imgs = torch.stack(imgs, dim=0)
        captionss = torch.LongTensor(captionss)

        return imgs, captionss