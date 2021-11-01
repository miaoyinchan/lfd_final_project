#!/bin/bash
#sbatch --time=03:00:00 --partition=gpu --gres=gpu:1 run.sh

module load cuDNN

cd src
python3 train.py -s 42
python3 test.py -s 42
python3 evaluate.py
