#!/usr/bin/env bash

set -e

cd src

python3 -m spacy download en_core_web_sm
python3 aug_train.py -b
python3 test.py -u1 -ts 25
python3 evaluate.py -u1

