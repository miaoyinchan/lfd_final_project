#!/usr/bin/env bash

set -e


python3 train.py  -bi -lay 2  -l 0.1 -e glove -m aug_model -b 16 -d 0.3 -i train_aug.csv -w
python3 test.py 
python3 evaluate.py 


