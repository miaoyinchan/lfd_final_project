#!/bin/bash

function help {
  
  
  echo -e "Usage:\n"
  
  echo -e "  testset: [mandatory argument] Use 24 to test the model on data from 24th meeting and 25 will test the model on 25th COP meeting"
   
  
  echo -e "  --option, [optional argument] use t to train, predict, and evaluate a model from scratch. by default it only predict outputs from a saved model and evaluate the result\n"
  
  echo "EXAMPLE: run.sh 25 t"
  
}


[ $# -eq 0 ] && { echo "Argument Not found"; help; exit 1; }

if [[ "$1" != "24" ]] && [[ "$1" != "25" ]]
then
 echo "Test Set Not Found"
 help
 exit 1;
fi

cd src
if [[ "$2" == "t" ]]
then
 python3 train.py
 python3 test.py -ts $1
 python3 evaluate.py

else
 python3 test.py -ts $1
 python3 evaluate.py
    

fi