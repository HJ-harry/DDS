#!/bin/bash

Nview=120
T_sampling=50
eta=0.85

python main.py \
    --type 3d \
    --config AAPM256.yml \
    --dataset_path "./indist_samples/CT/L067" \
    --Nview $Nview \
    --eta $eta \
    --deg "LA-CT" \
    --sigma_y 0.0 \
    --rho 10.0 \
    --lamb 0.4 \
    --T_sampling $T_sampling \
    -i ./results_3d