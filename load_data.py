from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PACSDatasetBaseline(Dataset):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.transform(self.data[index])
        y = self.label[index]
        return x, y

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    with open(f'{opt["data_path"]}/{source_domain}.txt') as f:
        source_lines = f.readlines()

    with open(f'{opt["data_path"]}/{target_domain}.txt') as f:
        target_lines = f.readlines()

    print(len(source_lines))
    print(len(target_lines))