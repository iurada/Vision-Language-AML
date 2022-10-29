import os
import logging
from parse_args import parse_arguments
from models.base_model import BaselineModel

def train(model, loader):
    pass

def evaluate():
    pass

def main(opt):
    if opt['experiment'] == 'baseline':
        model = BaselineModel()
    else:
        raise ValueError('Experiment not yet supported.')

if __name__ == '__main__':
    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='w')
    logging.info(opt)

    main(opt)