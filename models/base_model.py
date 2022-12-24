import torch
import torch.nn.functional as F
import torch.nn as nn
import components as cmp

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = cmp.FeatureExtractor()
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
        self.feature_extractor = cmp.FeatureExtractor()
        self.disentangler = cmp.Disentangler()
        self.category_classifier = cmp.CategoryClassifier()
        self.domain_classifier = cmp.DomainClassifier()
        self.reconstructor = cmp.Reconstructor()

    def forward(self, x, state=None, train=True):
        if state == 'category_disentanglement_phase_1' and train == True:
            # Loss for category classifier to minimize
            cmp.set_requires_grad(self.feature_extractor, requires_grad=True)
            cmp.set_requires_grad(self.disentangler, requires_grad=True)
            cmp.set_requires_grad(self.category_classifier, requires_grad=True)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=False) # Category encoder
            x = self.category_classifier(x)
            return x
        elif state == 'category_disentanglement_phase_1' and train == False:
            # Validation
            cmp.set_requires_grad(self.feature_extractor, requires_grad=False)
            cmp.set_requires_grad(self.disentangler, requires_grad=False)
            cmp.set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=False) # Category encoder
            x = self.category_classifier(x)
            return x
        elif state == 'domain_disentanglement_phase_1' and train == True:
            # Loss for domain classifier to minimize
            cmp.set_requires_grad(self.feature_extractor, requires_grad=True)
            cmp.set_requires_grad(self.disentangler, requires_grad=True)
            cmp.set_requires_grad(self.domain_classifier, requires_grad=True)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=True) # Domain encoder
            x = self.domain_classifier(x)
            return x
        elif state == 'domain_disentanglement_phase_1' and train == False:
            # Validation
            cmp.set_requires_grad(self.feature_extractor, requires_grad=False)
            cmp.set_requires_grad(self.disentangler, requires_grad=False)
            cmp.set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=True) # Domain encoder
            x = self.domain_classifier(x)
            return x
        elif state == 'category_disentanglement_phase_2' and train == True:
            # Loss for domain classifier to maximise
            cmp.set_requires_grad(self.feature_extractor, requires_grad=True)
            cmp.set_requires_grad(self.disentangler, requires_grad=True)
            cmp.set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=False) # Category encoder
            x = self.domain_classifier(x)
            return x
        elif state == 'category_disentanglement_phase_2' and train == False:
            # Validation
            cmp.set_requires_grad(self.feature_extractor, requires_grad=False)
            cmp.set_requires_grad(self.disentangler, requires_grad=False)
            cmp.set_requires_grad(self.domain_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=False) # Category encoder
            x = self.domain_classifier(x)
            return x
        elif state == 'domain_disentanglement_phase_2' and train == True:
            # Loss for category classifier to maximise
            cmp.set_requires_grad(self.feature_extractor, requires_grad=True)
            cmp.set_requires_grad(self.disentangler, requires_grad=True)
            cmp.set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=True) # Domain encoder
            x = self.category_classifier(x)
            return x
        elif state == 'domain_disentanglement_phase_2' and train == False:
            # Validation
            cmp.set_requires_grad(self.feature_extractor, requires_grad=False)
            cmp.set_requires_grad(self.disentangler, requires_grad=False)
            cmp.set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=True) # Domain encoder
            x = self.category_classifier(x)
            return x
        elif state == "feature_reconstruction" and train == True:
            # Loss for reconstructor to minimize
            cmp.set_requires_grad(self.feature_extractor, requires_grad=True)
            cmp.set_requires_grad(self.disentangler, requires_grad=True)
            cmp.set_requires_grad(self.reconstructor, requires_grad=True)
            x = self.feature_extractor(x)
            fcs = self.disentangler(x, domain=False)
            fds = self.disentangler(x, domain=True)
            rec = self.reconstructor(torch.cat(fcs, fds))
            return x, rec
        elif state == "feature_reconstruction" and train == False:
            # Validation
            cmp.set_requires_grad(self.feature_extractor, requires_grad=False)
            cmp.set_requires_grad(self.disentangler, requires_grad=False)
            cmp.set_requires_grad(self.reconstructor, requires_grad=False)
            x = self.feature_extractor(x)
            fcs = self.disentangler(x, domain=False)
            fds = self.disentangler(x, domain=True)
            rec = self.reconstructor(torch.cat(fcs, fds))
            return x, rec
        elif state == None:
            # Testing part
            cmp.set_requires_grad(self.feature_extractor, requires_grad=False)
            cmp.set_requires_grad(self.disentangler, requires_grad=False)
            cmp.set_requires_grad(self.category_classifier, requires_grad=False)
            x = self.feature_extractor(x)
            x = self.disentangler(x, domain=False)
            x = self.category_classifier(x)
            return x