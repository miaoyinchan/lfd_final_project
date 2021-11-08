#!/bin/bash

python3 -m spacy download en_core_web_sm

if [[ "$1" == "upsampling" ]]
then
 python3 data-processing.py
 echo data processing: finished
 python3 data-cleaning.py
 echo data cleaning: finished
 python3 data-resample.py -u -d
 echo data resampling: finished


else
 python3 data-processing.py
 echo data processing: finished
 python3 data-cleaning.py
 echo data cleaning: finished
 python3 data-resample.py -d
 echo data resampling: finished


fi