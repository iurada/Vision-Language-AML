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

    def forward(self, x, state, train, domain_generalization): #domain_generalization bandiera per punto 5

        if train == True:
            if state == 0:
                f = self.feature_extractor(x)
                fcs = self.category_encoder(f)
                cc = self.category_classifier(fcs)
                return cc
            elif state == 1:
                f = self.feature_extractor(x)
                fcs = self.category_encoder(f)
                dd = self.domain_classifier(fcs, domain_generalization)
                return dd
            elif state == 2:
                f = self.feature_extractor(x)
                fds = self.domain_encoder(f)
                dd = self.domain_classifier(fds, domain_generalization)
                return dd
            elif state == 3:
                f = self.feature_extractor(x)
                fds = self.domain_encoder(f)
                cc = self.category_classifier(fds)
                return cc
            elif state == 4:
                f = self.feature_extractor(x)
                fds = self.domain_encoder(f)
                fcs = self.category_encoder(f)
                r = self.reconstructor(torch.cat((fds, fcs), 1))
                return r, f
        else:
            f = self.feature_extractor(x)
            fcs = self.category_encoder(f)
            cc = self.category_classifier(fcs)
            return cc        

class CLIPDisentangleModel(nn.Module):
    def __init__(self):
        super(CLIPDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder= CategoryEncoder()
        self.domain_encoder = DomainEncoder()
        self.category_classifier = CategoryClassifier()
        self.domain_classifier = DomainClassifier()
        self.reconstructor = Reconstructor()

    def forward(self, x, state, train, domain_generalization): #domain_generalization bandiera per punto 5

        if train == True:
            if state == 0:
                f = self.feature_extractor(x)
                fcs = self.category_encoder(f)
                cc = self.category_classifier(fcs)
                return cc
            elif state == 1:
                f = self.feature_extractor(x)
                fcs = self.category_encoder(f)
                dd = self.domain_classifier(fcs, domain_generalization)
                return dd
            elif state == 2:
                f = self.feature_extractor(x)
                fds = self.domain_encoder(f)
                dd = self.domain_classifier(fds, domain_generalization)
                return dd
            elif state == 3:
                f = self.feature_extractor(x)
                fds = self.domain_encoder(f)
                cc = self.category_classifier(fds)
                return cc
            elif state == 4:
                f = self.feature_extractor(x)
                fds = self.domain_encoder(f)
                fcs = self.category_encoder(f)
                r = self.reconstructor(torch.cat((fds, fcs), 1))
                return r, f
        else:
            f = self.feature_extractor(x)
            fcs = self.category_encoder(f)
            cc = self.category_classifier(fcs)
            return cc            
