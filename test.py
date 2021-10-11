from one_utils import eval_torch_acc
from torchvision.models import alexnet
model = alexnet(pretrained=True)

data_path = "/DATA/disk1/ImageNet/extract"
eval_torch_acc(model, data_path, gpus="1")