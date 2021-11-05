#!/bin/bash


cd src
if [[ "$1" == "train" ]]
then
 python3 train.py
 python3 test.py
 python3 evaluate.py

elif [[ "$1" == "test" ]]
then
 python3 test.py
 python3 evaluate.py
    
elif [[ "$1" == "eval" ]]
then
 python3 evaluate.py

else
    echo "Please include the type of experiment (train, test, eval) or check spelling"

fi