#!/bin/bash

python3.7 src/main.py train --train-path data2/train2.pid \
--dev-path data2/dev2.pid \
--model-path-base models/bnp_phobertlarge \
--use-pretrained \
--pretrained-model vinai/phobert-large \
--numpy-seed 42 \
--use-encoder
