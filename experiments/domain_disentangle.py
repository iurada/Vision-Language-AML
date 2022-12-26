import torch
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion_1 = torch.nn.CrossEntropyLoss()
        self.criterion_2 = torch.nn.MSELoss()

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

    def train_iteration(self, data, state):
        x, y, domain = data
        x = x.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)

        logits = self.model(x, state, train=True)

        if state == 'category_disentanglement_phase_1':
            loss = self.criterion_1(logits, y) # Minimize loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif state == 'domain_disentanglement_phase_1':
            loss = self.criterion_1(logits, domain) # Minimize loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif state == 'category_disentanglement_phase_2':
            loss = -self.criterion_1(logits, y) # Maximize loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif state == 'domain_disentanglement_phase_2':
            loss = -self.criterion_1(logits, domain) # Maximize loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif state == "feature_reconstruction":
            loss = self.criterion_1(logits[0], logits[1]) # Minimize loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()


    def validate(self, loader, state):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y, domain in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                domain = domain.to(self.device)

                logits = self.model(x, state, train=False)

                if state == 'category_disentanglement_phase_1':
                    loss += self.criterion(logits, y)
                    pred = torch.argmax(logits, dim=-1)
                    accuracy += (pred == y).sum().item()
                    count += x.size(0)
                elif state == 'domain_disentanglement_phase_1':
                    loss += self.criterion(logits, domain)
                    pred = torch.argmax(logits, dim=-1)
                    accuracy += (pred == domain).sum().item()
                    count += x.size(0)
                elif state == 'category_disentanglement_phase_2':
                    loss += self.criterion(logits, y)
                    pred = torch.argmax(logits, dim=-1)
                    accuracy += (pred == y).sum().item()
                    count += x.size(0)
                elif state == 'domain_disentanglement_phase_2':
                    loss += self.criterion(logits, domain)
                    pred = torch.argmax(logits, dim=-1)
                    accuracy += (pred == domain).sum().item()
                    count += x.size(0)
                elif state == 'feature_reconstruction':
                    loss += self.criterion(logits[0], logits[1])
                    pred = torch.argmax(logits[1], dim=-1)
                    accuracy += (pred == logits[0]).sum().item()
                    count += x.size(0)
                elif state == None:
                    loss += self.criterion(logits, y)
                    pred = torch.argmax(logits, dim=-1)
                    accuracy += (pred == y).sum().item()
                    count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss