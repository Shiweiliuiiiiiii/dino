#!/bin/bash
#SBATCH --job-name=linear_slak_3x3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o linear_slak_3x3_S_R_A_300.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
#source activate slak
cd ../../

ck1=/datadrive_c/ssl/110epochs/dino_slak_3_bn/checkpoint.pth
ck2=/datadrive_c/ssl/110epochs/dino_slak_3_bn/linear/checkpoint.pth.tar
path=/datadrive_c/ImageNet-C/
python -m torch.distributed.launch --nproc_per_node=1 --master_port=11111  eval_linear.py --evaluate --data_path $path --dataset imagenet_c \
--arch SLaK_tiny --kernel_size 3 3 3 3 100 --LoRA False --bn True \
--pretrained_weights $ck1 --url $ck2 \
--output_dir ./ --checkpoint_key teacher

#python  -m torch.distributed.launch --nproc_per_node=1 --master_port=54444  eval_linear.py --evaluate --data_path /home/sliu/project_space/datasets/sketch \
#--arch SLaK_tiny --kernel_size 3 3 3 3 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher

#python  -m torch.distributed.launch --nproc_per_node=1 --master_port=54444  eval_linear.py --evaluate --data_path /home/sliu/project_space/datasets/imagenet-r \
#--arch SLaK_tiny --kernel_size 3 3 3 3 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher

#python  -m torch.distributed.launch --nproc_per_node=1 --master_port=54444  eval_linear.py --evaluate --data_path /home/sliu/project_space/datasets/imagenet-a \
#--arch SLaK_tiny --kernel_size 3 3 3 3 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher

source deactivate
