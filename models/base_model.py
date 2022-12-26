import torch
import torch.nn.functional as F
import torch.nn as nn
from components import * 

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
        if state == 'category_disentanglement_phase_1' and train == True:
            # Loss for category classifier to minimize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.category_classifier, requires_grad=True)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=False) # Category encoder
            x = self.category_classifier(x)
            return x
        if state == 'category_disentanglement_phase_1' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=False) # Category encoder
            x = self.category_classifier(x)
            return x

        '''Train and validation parts for domain encoder's first step'''
        if state == 'domain_disentanglement_phase_1' and train == True:
            # Loss for domain classifier to minimize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=True)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=True) # Domain encoder
            x = self.domain_classifier(x)
            return x
        if state == 'domain_disentanglement_phase_1' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=True) # Domain encoder
            x = self.domain_classifier(x)
            return x

        '''Train and validation parts for category encoder's second step'''
        if state == 'category_disentanglement_phase_2' and train == True:
            # Loss for domain classifier to maximize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=False) # Category encoder
            x = self.domain_classifier(x)
            return x
        if state == 'category_disentanglement_phase_2' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.disentangler, requires_grad=False)
            set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=False) # Category encoder
            x = self.domain_classifier(x)
            return x
        
        '''Train and validation parts for domain encoder's second step'''
        if state == 'domain_disentanglement_phase_2' and train == True:
            # Loss for category classifier to maximize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=True) # Domain encoder
            x = self.category_classifier(x)
            return x
        if state == 'domain_disentanglement_phase_2' and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.encoder(x, domain=True) # Domain encoder
            x = self.category_classifier(x)
            return x

        '''Train and validation parts for reconstructor'''
        if state == "feature_reconstruction" and train == True:
            # Loss for reconstructor to minimize
            set_requires_grad(self.feature_extractor, requires_grad=True)
            set_requires_grad(self.encoder, requires_grad=True)
            set_requires_grad(self.reconstructor, requires_grad=True)
            x = self.feature_extractor(x)
            fcs = self.encoder(x, domain=False)
            fds = self.encoder(x, domain=True)
            rec = self.reconstructor(torch.cat(fcs, fds))
            return x, rec
        if state == "feature_reconstruction" and train == False:
            # Validation
            set_requires_grad(self.feature_extractor, requires_grad=False)
            set_requires_grad(self.encoder, requires_grad=False)
            set_requires_grad(self.reconstructor, requires_grad=False)
            x = self.feature_extractor(x)
            fcs = self.encoder(x, domain=False)
            fds = self.encoder(x, domain=True)
            rec = self.reconstructor(torch.cat(fcs, fds))
            return x, rec

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