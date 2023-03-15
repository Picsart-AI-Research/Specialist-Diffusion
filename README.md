# Specialist-Diffusion
Specialist Diffusion: Extremely Low-Shot Fine-Tuning of Large Diffusion Models

## Setup the environment
First, install prerequisites with:

    conda env create -f environment.yml
    conda activate sd
  
Then, set up the configuration for accelerate with:

    accelerate config

## Train a model
If you never used huggingface before, run

    huggingface-cli login
    
so the pretrained weights will be automatically downloaded.

An example call:

    accelerate launch train.py --config='configs/train_default.json'

## Evaluate a model
An example call:

    accelerate launch eval.py --config='configs/eval_default.json'
