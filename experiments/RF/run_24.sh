#!/usr/bin/env bash

set -e

cd src

python3 train.py -t -b
python3 test.py -t -b -ts 24
python3 evaluate.py -t -b

