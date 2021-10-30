#!/bin/bash
#sbatch --time=03:00:00 --partition=gpu --gres=gpu:1 run.sh

module load cuDNN

cd src
python3 train.py -t
python3 test.py -t
python3 evaluate.py -t
