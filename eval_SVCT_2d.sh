#!/bin/bash

Nview=8
T_sampling=50
eta=0.85

python main.py \
    --type '2d' \
    --config AAPM256.yml \
    --dataset_path "./indist_samples/CT/L067" \
    --Nview $Nview \
    --eta $eta \
    --deg "SV-CT" \
    --sigma_y 0.01 \
    --T_sampling 100 \
    --T_sampling $T_sampling \
    -i ./results