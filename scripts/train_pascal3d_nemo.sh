#!/usr/bin/env bash

gpu=0
pretrain="pretrain_models/res50_moco_v2_800ep_pretrain.pth"
# pretrain=""
exp_dir="exps/$1.PoseContrast_Pascal3D_NeMo_MOCO_v2"
# exp_dir="exps/$1.PoseContrast_Pascal3D_NeMo_ImageNetk1_v2"

python src/train.py --gpu $gpu --dataset Pascal3D --nemo_mode --out ${exp_dir} \
    --bs 32 --epochs 15 --lr_step 12 --weighting linear  --poseNCE 1 --tau 0.5 --crop --pretrain ${pretrain}
