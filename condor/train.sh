#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dml

dir=/users/visics/gkouros/projects/pose-estimation/PoseContrast
cd $dir

./scripts/train_pascal3d_nemo.sh

conda deactivate
