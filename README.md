# OneUtils
A collection of useful tools for oneflow

## Installation
```bash
pip install one-utils==0.0.2
```

## Usage
- Weight Transfer
```python
from one_utils import convert_torch_to_flow
model = model()
torch_weight = "path/to/pytorch_pretrained_weight.pth"
save_path = "path/to/save"
convert_torch_to_flow(model, torch_weight_path, save_path)
```

- Eval Model Acc on ImageNet
```python
from one_utils import eval_torch_acc
model = model()
data_path = "path/to/imagenet/val"
eval_torch_acc(model, data_path, n_gpu_use=1)
```