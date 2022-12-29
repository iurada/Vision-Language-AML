import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = x.squeeze()
        if len(x.size()) < 2:
            x = x.unsqueeze(0)
        return x

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        b = b/b.size()[1]
        return b

class CategoryEncoder(nn.Module):
    def __init__(self):
        super(CategoryEncoder, self).__init__()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.category_encoder(x)
        return x

class DomainEncoder(nn.Module):
    def __init__(self):
        super(DomainEncoder, self).__init__()
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.domain_encoder(x)
        return x

class CategoryClassifier(nn.Module):
    def __init__(self):
        super(CategoryClassifier, self).__init__()
        self.category_classifier = nn.Linear(32, 7)
    
    def forward(self, x):
        x = self.category_classifier(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.domain_classifier(x)
        return x

class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),

            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),

            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
        )

    def forward(self, x):
        x = self.feature_reconstructor(x)
        return x
