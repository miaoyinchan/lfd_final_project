#!/usr/bin/env bash

set -e

python3 train.py -k linear
python3 test.py 
python3 evaluate.py


