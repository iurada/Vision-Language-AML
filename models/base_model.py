import torch.nn as nn
from torchvision.models import resnet18

class BaselineModel(nn.Module):

    def __init__(self):
        super(BaselineModel, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        print(self.resnet18)

        