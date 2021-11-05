#!/usr/bin/env bash

set -e

cd src

python3 aug_train.py -b
python3 test.py -u1
python3 evaluate.py -u1

