#!/bin/bash

cd ~/Projects/Sonar/sonar-analysis/ipynb/Script/bsub

source ~/.virtualenv/sonar/bin/activate
source ~/Projects/Sonar/sonar-analysis/setup.sh

# Run python code
python $1 $2 $3



