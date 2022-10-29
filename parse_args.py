import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='baseline', 
        choices=['baseline', 'domain_disentangle', 'clip_disentangle', 'domain_generalization', 'finetuned_clip'])

    parser.add_argument('--target_domain', type=str, default='cartoon', choices=['cartoon', 'sketch', 'photo'])

    parser.add_argument('--output_path', type=str, default='.', help='Where to create the output directory containing logs and weights.')
    parser.add_argument('--data_path', type=str, default='data/PACS', help='Locate the PACS dataset on disk.')
    

    #! Additional arguments can go below this line:
    #parser.add_argument('--test', type=str, default='some default value', help='some hint that describes the effect')

    opt = vars(parser.parse_args())

    opt['output_path'] = f'{opt["output_path"]}/record/{opt["experiment"]}'

    return opt