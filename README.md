# Show, attend and tell 구현

[Show, attend and tell](https://arxiv.org/abs/1502.03044) 논문 구현

## 배운 점

무언가를 밑바닥부터 구현하는 프로젝트가 처음이다보니 좀 버벅거리는 부분이 많았다. (덕분에 배운 점도 많다!)

- 모듈화  
  `models.py`, `utils.py`, `train.py`, `eval.py` 등 목적에 맞게 모든 파일을 분리하는 연습을 했다.  
  처음에는 `main.py`에 train, eval 등을 모두 때려박으려 했는데 의외로 코드가 분리되지 않았고,  
  참고용 코드를 보니 파일을 분리해놨길래 나도 분리해봤다.

  파일을 분리하는 기준을 나름 알아보면 좋을 것 같다.

- google docstring convention  
  주석을 잘 달아서 가독성 좋은 코드를 짜는 게 항상 중요하다고 생각했었다.  
  주석 다는 규칙이 있나 요기조기 찾다보니 [google docstring convention](https://google.github.io/styleguide/pyguide.html)을 찾았다.  
  너무 길어서 다 읽어보진 못했는데, 중요해 보이는 몇 가지 포인트는 코드에 최대한 반영해보려 노력했다.

  추가로, torch.Tensor를 다루는 코드의 경우  
  Tensor.shape를 주석에 명시해주면 서로 꼬이는 것을 방지할 수 있어  
  습관화하는 게 좋을 듯 하다.

- torch 문법들  
  - DataLoader parameters
    [링크](https://subinium.github.io/pytorch-dataloader/)가 큰 도움이 되었다.
    `collate_fn`: COCO는 사진마다 크기가 달라 필수라 생각했다.
    `drop_last`: 마지막에 남은 데이터가 1개짜리면 batch normalization에서 에러가 난다. `True`로 설정했다.
    `pin_memory`: `resize`와 `collate_fn`을 통해 고정된 크기의 텐서가 GPU로 전달될 것이기 때문에 `True`로 설정해주는 게 좋을 것 같다.

- collections.Counter  
  collections.defaultdict를 배울 때도 그랬듯이,  
  전혀 예상치 못하게 배운 기능이다.  
  사용법이 어려운 건 아니니 여기서는 추가적인 설명은 적지 않아도 될 듯.
  

## 논문 내 변수들
- `K` : 단어 임베딩 벡터의 크기 (= `embed_size`)
- `D` : 각 픽셀의 정보를 담는 벡터의 크기. CNN을 통과한 결과물의 채널 (= `feature_c`)
- `L` : 추출된 픽셀의 갯수 (= `n_pixels` = `feature_w` * `feature_h`)
- `m` : embedding 크기 (= `decoder_in`) # 그럼 얘는 K랑 다른가?
- `n` : LSTM dimensionality (= `decoder_hidden`)
- `z` : context vector (= `context`)