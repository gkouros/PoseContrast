#!/usr/bin/env bash

gpu=0
# exp_dir="exps/PoseContrast_Pascal3D_MOCOv2"
exp_dir="exps/13452.PoseContrast_Pascal3D_NeMo_MOCO_v2"

python src/test.py --gpu $gpu --dataset KITTI3D --ckpt ${exp_dir}/ckpt.pth --nemo_format --occlusion_level $1