#!/usr/bin/env bash

gpu=0
# exp_dir="exps/PoseContrast_Pascal3D_MOCOv2"
# exp_dir="exps/PoseContrast_Pascal3D_NeMo_ImageNetk1_v2"
exp_dir="exps/13452.PoseContrast_Pascal3D_NeMo_MOCO_v2"
# exp_dir="exps/13456.PoseContrast_Pascal3D_NeMo_MOCO_v2_no_crop/"
# exp_dir="exps/13455.PoseContrast_Pascal3D_NeMo_ImageNetk1_v2_more_epochs"

python src/test.py --gpu $gpu --dataset Pascal3D --ckpt ${exp_dir}/ckpt.pth --crop --nemo_format --occlusion_level $1
# python src/test.py --gpu $gpu --dataset Pascal3D --ckpt ${exp_dir}/ckpt.pth --nemo_format --occlusion_level $1
# python src/test.py --gpu $gpu --dataset Pascal3D --ckpt ${exp_dir}/ckpt.pth --crop --occlusion_level $1