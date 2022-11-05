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

ck1=/datadrive_c/ssl/300epochs/slak_small_300_240/checkpoint.pth
ck2=/datadrive_c/ssl/300epochs/slak_small_300_240/linear/checkpoint.pth.tar
path=/datadrive_c/Imagenet-C/
python -m torch.distributed.launch --nproc_per_node=1 --master_port=11115  eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
--arch SLaK_small --kernel_size 9 9 9 9 100 --LoRA False --bn True \
--pretrained_weights $ck1 --url $ck2 \
--output_dir ./ --checkpoint_key teacher
#



source deactivate
