#!/bin/bash
#sbatch --time=02:00:00 --partition=gpu --gres=gpu:1 run.sh

module load cuDNN

cd src
python3 train.py
python3 test.py
python3 evaluate.py
