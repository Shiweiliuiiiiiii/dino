#!/bin/bash

source /home/sliu/miniconda3/etc/profile.d/conda.sh
source activate slak

python run_with_submitit.py --nodes 2 --ngpus 4 --bn True --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false \
--arch SLaK_small --kernel_size 9 9 9 9 100 --epochs 300 --batch_size_per_gpu 128 --timeout 2  \
--data_path /projects/2/managed_datasets/imagenet/train --output_dir /projects/0/prjste21060/projects/dino/dino_slak_small_9_bn_300_2nodes/

source deactivate