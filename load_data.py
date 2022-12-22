from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}


class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

def read_lines(data_path, domain_name):
    examples = {}
    with open(f'./Vision-Language-AML/{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        main_folder = './Vision-Language-AML/'
        local_folder = '/kfold/'
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        # image_path_zzz = f'./Vision-Language-AML/{data_path}/{domain_name}/{category_name}/{image_name}'
        # print(f'{line}, image_path_zzz {image_path_zzz}, image_name {image_name}, category_name {category_name}, category_idx {category_idx}')
        image_path = f'{main_folder}{data_path}{local_folder}{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
        
    return examples

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


class PACSDatasetDomainDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, domain

def build_splits_domain_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    print(f'Source cat len {[(k, len(v)) for k,v in source_examples.items()]}')
    print(f'Target cat len {[(k, len(v)) for k,v in target_examples.items()]}')

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    source_val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    tot_source = sum([len(v) for k,v in source_examples.items()])
    tot_target = sum([len(v) for k,v in target_examples.items()])

    domain_ratios = {
        0: tot_source/(tot_source+tot_target),
        1: tot_target/(tot_source+tot_target),
    } 

    print(f'Dictionary {source_category_ratios}')
    print(f'Dictionary {domain_ratios}')
    
    domain_val_split_length = (tot_source+tot_target)*0.2

    train_examples_source = []
    val_examples_source = []
    train_examples_target = []
    train_examples_s = []
    train_examples_t = []
    val_examples_target = []
    test_examples = []

    # To change from dictionary (key: category, value: paths) to a list of paths
    for category, example_list in source_examples.items():
        for example in example_list:
            train_examples_s.append([example, category, 0])
    
    # To change from dictionary (key: category, value: paths) to a list of paths
    for category, example_list in target_examples.items():
        for example in example_list:
            train_examples_t.append([example, category, 1])
            test_examples.append([example, category, 1])

    # Union of the two domains
    train_examples_t += train_examples_s

    for i, parameters in enumerate(train_examples_s):
        split_idx = round(source_category_ratios[parameters[1]] * source_val_split_length)
        if i > split_idx:
            train_examples_source.append([parameters[0], parameters[1], parameters[2]])
        else:
            val_examples_source.append([parameters[0], parameters[1], parameters[2]])
    
    for i, parameters in enumerate(train_examples_t):
        split_idx = round(domain_ratios[parameters[2]] * domain_val_split_length)
        if i > split_idx:
            train_examples_target.append([parameters[0], parameters[1], parameters[2]])
        else:
            val_examples_target.append([parameters[0], parameters[1], parameters[2]])

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader_1 = DataLoader(PACSDatasetDomainDisentangle(train_examples_source, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True) 
    train_loader_2 = DataLoader(PACSDatasetDomainDisentangle(train_examples_target, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader_1 = DataLoader(PACSDatasetDomainDisentangle(val_examples_source, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    val_loader_2 = DataLoader(PACSDatasetDomainDisentangle(val_examples_target, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader_1, train_loader_2, val_loader_1, val_loader_2, test_loader

def build_splits_clip_disentangle(opt):
    raise NotImplementedError('[TODO] Implement build_splits_clip_disentangle') #TODO
