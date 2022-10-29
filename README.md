# AML 2022/2023 Project - Vision and Language
Official repository for the "Vision and Language" project - Advanced Machine Learning 2022/2023 @ PoliTo

## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.1, installed and correctly configured on your system.
To ensure full compatibility, make sure to use Python version 3.9.13.

Once you have properly setup everything, make sure you are in the correct directory and run from the command line:
```bash
pip install -r requirements.txt
```

### Dataset
1 - Download PACS dataset from the portal of the course in the "project_topics" folder.

2 - Place the dataset in the 'data/PACS' folder making sure that the images are organized in this way:
```
data/PACS/kfold/art_painting/dog/pic_001.jpg
data/PACS/kfold/art_painting/dog/pic_002.jpg
data/PACS/kfold/art_painting/dog/pic_003.jpg
...
```

At this point you should be able to run and edit the base code provided.

## Code Structure
- `main.py`: main entry point. Check out below which command line arguments can be passed when running an experiment via:
    ```bash
    python main.py
    ```
- `parse_args.py`: contains the function responsible for parsing each command line argument.
- `load_data.py`: contains the code to load data, build splits and dataloaders.

## Command Line Arguments
- `--experiment`: which experiment to run chosen among the following:
  - `baseline`: runs the experiment needed to reproduce the baseline (see point 1. of the project)
  - `domain_disentangle`: runs the domain disentanglement experiment (see point 2. of the project)
  - `clip_disentangle`: runs the disentanglement experiment using CLIP (see point 4. of the project)
  - `domain_generalization`: runs the domain generalization experiment (see Variation 1 of the project)
  - `finetuned_clip`: runs the disentanglement experiment using the finetuned version of CLIP (see Variation 2 of the project and check other arguments to see how to load the finetuned CLIP weights)

## Baseline Results (see point 1. of the project)
TODO