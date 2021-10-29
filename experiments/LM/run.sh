#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module load cuDNN

cd src
python3 train.py -t
python3 test.py -t
python3 evaluate.py -t
