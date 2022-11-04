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

#source /home/sliu/miniconda3/etc/profile.d/conda.sh
#source activate slak
#cd ../../

#ck1=/datadrive_c/ssl/100epochs/dino_slak_3_bn/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_slak_3_bn/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11111  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch SLaK_tiny --kernel_size 3 3 3 3 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher
#
#
#ck1=/datadrive_c/ssl/100epochs/dino_slak_5_bn/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_slak_5_bn/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11112  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch SLaK_tiny --kernel_size 5 5 5 5 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher
#
#
#ck1=/datadrive_c/ssl/100epochs/dino_slak_7_bn/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_slak_7_bn/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11113  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch SLaK_tiny --kernel_size 7 7 7 7 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher
#
#
#ck1=/datadrive_c/ssl/100epochs/dino_slak_9_bn/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_slak_9_bn/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11114  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch SLaK_tiny --kernel_size 9 9 9 9 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher
#
#
#ck1=/datadrive_c/ssl/100epochs/dino_slak_15_bn/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_slak_15_bn/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11115  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch SLaK_tiny --kernel_size 15 15 15 15 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher
#
#
#ck1=/datadrive_c/ssl/100epochs/dino_slak_15_bn/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_slak_15_bn/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11116  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch SLaK_tiny --kernel_size 15 15 15 15 100 --LoRA False --bn True \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher
#
#
ck1=/datadrive_c/ssl/100epochs/dino_swin_tiny/checkpoint.pth
ck2=/datadrive_c/ssl/100epochs/dino_swin_tiny/linear/checkpoint.pth.tar
path=/datadrive_c/Imagenet-C/
python -m torch.distributed.launch --nproc_per_node=1 --master_port=11117  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
--arch swin_tiny \
--cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml \
--pretrained_weights $ck1 --url $ck2 \
--output_dir ./ --checkpoint_key teacher \
--batch_size_per_gpu 256 --n_last_blocks 4 --num_labels 1000 MODEL.NUM_CLASSES 0


#ck1=/datadrive_c/ssl/100epochs/dino_RN50/checkpoint.pth
#ck2=/datadrive_c/ssl/100epochs/dino_RN50/linear/checkpoint.pth.tar
#path=/datadrive_c/Imagenet-C/
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=11118  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
#--arch resnet50 \
#--pretrained_weights $ck1 --url $ck2 \
#--output_dir ./ --checkpoint_key teacher \
#--batch_size_per_gpu 256


source deactivate
