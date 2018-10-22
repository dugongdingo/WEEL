#!/usr/bin/env bash

LEARNING_RATES=("0.01" "0.001" "0.0001" "0.00001" "0.1")

DROPOUTS=("0.01" "0.001" "0.0001" "0.00001" "0.1")

for lr in ${LEARNING_RATES[@]}; do
  for d in ${DROPOUTS[@]}; do
    python3 -m weel --learningrate $lr --dropout $d
  done
done
