import torch
import torch.nn as nn
import clip
from models.base_model import CLIPDisentangleModel
from models.components import *

class CLIPDisentangleExperiment:
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = CLIPDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup CLIP model
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False    #to freeze the clip model

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion_CEL = torch.nn.CrossEntropyLoss(ignore_index=42)     #ingnore index 42 that is put to discriminate the target
        self.criterion_MSEL = torch.nn.MSELoss()
        self.criterion_EL = EntropyLoss()

        # Setup loss weights
        self.weights = [3, 1.5, 0.5, 1] # 2 2 1
        self.alpha_cat = 0.05
        self.alpha_dom = 0.003

        # Set Domain Generalization
        self.domain_generalization = opt['dom_gen']

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, train): #train flag se false per validation/test
        x, desc, y, domain = data
        
        x = x.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)

        logits = self.model(x, train, self.domain_generalization)
        # logits[0] CE+CC
        # logits[1] DE+DC
        # logits[2] DE+CC
        # logits[3] CE+DC
        # logits[4] R
        # logits[5] F
        # logits[6] FDS
        
        tokenized_desc = clip.tokenize(desc).to(self.device)
        logit_clip = self.clip_model.encode_text(tokenized_desc)
        # logit_clip = features generate from text

        loss_class_1 = self.criterion_CEL(logits[0], y) # CEL of the categories
        loss_class_2 = self.alpha_cat*(-self.criterion_EL(logits[3]))
        loss_class = loss_class_1 + loss_class_2 # Category encoder

        loss_domain_1 = self.criterion_CEL(logits[1], domain) # CEL of the domains
        loss_domain_2 = self.alpha_dom*(-self.criterion_EL(logits[2]))
        loss_domain = loss_domain_1 + loss_domain_2 # Domain encoder

        loss_reconstructor = self.criterion_MSEL(logits[4], logits[5]) # Reconstructor

        loss_clip = self.criterion_MSEL(logit_clip, logits[6])

        loss = self.weights[0]*loss_class + self.weights[1]*loss_domain + self.weights[2]*loss_reconstructor + self.weights[3]*loss_clip

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, loader, train):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, desc, y, domain in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                domain = domain.to(self.device)

                logits = self.model(x, train, False)

                loss = self.weights[0]*self.criterion_CEL(logits, y) # Category encoder + Category classifier

                pred = torch.argmax(logits, dim=-1)
                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss