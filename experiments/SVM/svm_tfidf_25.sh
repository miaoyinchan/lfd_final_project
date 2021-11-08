#!/usr/bin/env bash

set -e

python3 train.py -k linear -t
python3 test.py -ts 25 -t
python3 evaluate.py -t




