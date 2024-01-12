#!/bin/bash
T_sampling=50

python main.py \
--type 2d \
--config fastmri_knee_320_complex.yml \
--dataset_path "./indist_samples/MRI/file1000033" \
--exp ./exp \
--ckpt_load_name "fastmri_knee_320_complex_1m.pt" \
--deg "MRI-multi" \
--sigma_y 0.01 \
--T_sampling $T_sampling \
-i ./results