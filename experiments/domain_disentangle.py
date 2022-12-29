import torch
import torch.nn as nn
from models.base_model import DomainDisentangleModel
from models.components import *

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion_CEL = torch.nn.CrossEntropyLoss(ignore_index=42)
        self.criterion_MSEL = torch.nn.MSELoss()
        self.criterion_EL = EntropyLoss()

        # Setup loss weights
        self.weights = [1, 2.591, 0.021, 0.594, 1.123]

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

    def train_iteration(self, data, train):
        x, y, domain = data
        x = x.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)

        logits = self.model(x, train)

        loss = 0
        
        loss += self.weights[0]*self.criterion_CEL(logits[0], y) # Category encoder + Category classifier
        loss += self.weights[1]*self.criterion_CEL(logits[1], domain) # Domain encoder + Domain classifier
        loss += self.weights[2]*self.criterion_EL(logits[2], y, True) # Domain encoder + Category classifier
        loss += self.weights[3]*self.criterion_EL(logits[3], domain, False) # Category encoder + Domain classifier
        loss += self.weights[4]*self.criterion_MSEL(logits[4], logits[5]) # Reconstructor

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
            for x, y, domain in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                domain = domain.to(self.device)

                logits = self.model(x, train)

                loss = self.weights[0]*self.criterion_CEL(logits, y) # Category encoder + Category classifier

                pred = torch.argmax(logits, dim=-1)
                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss