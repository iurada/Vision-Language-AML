
#TODO
class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def load_checkpoint(self, path):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def train_iteration(self, data):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def validate(self, loader):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')