
#TODO
class CLIPDisentangleExperiment: # See point 4. of the project
    
    def __init__(self, opt):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')

    def load_checkpoint(self, path):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')

    def train_iteration(self, data):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')

    def validate(self, loader):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')