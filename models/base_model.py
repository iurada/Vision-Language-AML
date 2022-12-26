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
        self.encoder = Encoder()
        self.category_classifier = CategoryClassifier()
        self.domain_classifier = DomainClassifier()
        self.reconstructor = Reconstructor()

    def forward(self, x, state=None, train=True):

        '''Train and validation parts for category encoder's first step'''
        if state == 'phase_1_category_disentanglement' and train == True:
            # Loss for category classifier to minimize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.category_classifier, requires_grad=True)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=False) # Category encoder
            x = self.category_classifier(x)
            return x
        if state == 'phase_1_category_disentanglement' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=False) # Category encoder
            x = self.category_classifier(x)
            return x

        '''Train and validation parts for domain encoder's first step'''
        if state == 'phase_1_domain_disentanglement' and train == True:
            # Loss for domain classifier to minimize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=True)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=True) # Domain encoder
            x = self.domain_classifier(x)
            return x
        if state == 'phase_1_domain_disentanglement' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=True) # Domain encoder
            x = self.domain_classifier(x)
            return x

        '''Train and validation parts for second steps'''
        if state == 'phase_2' and train == True:
            # Training
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.category_classifier, requires_grad=False)
            set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            fcs = self.encoder(x, domain=False) # Category encoder
            fds = self.encoder(x, domain=True) # Domain encoder
            dclf = self.domain_classifier(fcs)
            cclf = self.category_classifier(fds)
            rec = self.reconstructor(torch.cat((fds, fcs), 1))
            return dclf, cclf, rec, x
        if state == 'phase_2' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.category_classifier, requires_grad=False)
            set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            fcs = self.encoder(x, domain=False) # Category encoder
            fds = self.encoder(x, domain=True) # Domain encoder
            dclf = self.domain_classifier(fcs)
            cclf = self.category_classifier(fds)
            rec = self.reconstructor(torch.cat((fds, fcs), 1))
            return dclf, cclf, rec, x

        '''Test part'''
        if state == None:
            # Testing part
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=False)
            x = self.category_classifier(x)
            return x