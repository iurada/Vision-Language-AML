import torch
import torch.nn.functional as F
import torch.nn as nn
from models.components import * 

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder= CategoryEncoder()
        self.domain_encoder = DomainEncoder()
        self.category_classifier = CategoryClassifier()
        self.domain_classifier = DomainClassifier()
        self.reconstructor = Reconstructor()

    def forward(self, x, state=None):

        '''Train and validation parts for category encoder's first step'''
        if state == 'phase_1_category_disentanglement':
            # Loss for category classifier to minimize
            x = self.feature_extractor(x)
            x = self.category_encoder(x) 
            x = self.category_classifier(x)
            return x

        '''Train and validation parts for domain encoder's first step'''
        if state == 'phase_1_domain_disentanglement':
            # Loss for domain classifier to minimize
            x = self.feature_extractor(x)
            x = self.domain_encoder(x) # Domain encoder
            x = self.domain_classifier(x)
            return x

        '''Train and validation parts for second steps'''
        if state == 'phase_2':
            # Training
            set_requires_grad(self.category_classifier, requires_grad=False) # Freeze category classifier
            set_requires_grad(self.domain_classifier, requires_grad=False) # Freeze domain classifier
            x = self.feature_extractor(x)
            fcs = self.category_encoder(x) # Category encoder
            fds = self.domain_encoder(x) # Domain encoder
            dclf = self.domain_classifier(fcs)
            cclf = self.category_classifier(fds)
            rec = self.reconstructor(torch.cat((fds, fcs), 1))
            return cclf, dclf, rec, x

        '''Test part'''
        if state == None:
            # Testing part
            x = self.feature_extractor(x)
            x = self.category_encoder(x)
            x = self.category_classifier(x)
            return x