import torch
import clip
from models.base_model import DomainDisentangleModel

class CLIPDisentangleExperiment:
    
    def __init__(self, opt):
        pass

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        pass

    def load_checkpoint(self, path):
        pass

    def train_iteration(self, data):
        pass

    def validate(self, loader):
        pass