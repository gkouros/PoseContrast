#!/usr/bin/env bash

gpu=0
# exp_dir="exps/PoseContrast_Pascal3D_MOCOv2"
exp_dir="exps/PoseContrast_Pascal3D_NeMo_ImageNetk1_v2"

python src/test.py --gpu $gpu --dataset Pascal3D --ckpt ${exp_dir}/ckpt.pth --crop --nemo_format --occlusion_level 3