ck1=/datadrive_c/ssl/100epochs/dino_slak_3_bn/checkpoint.pth
ck2=/datadrive_c/ssl/100epochs/dino_slak_3_bn/linear/checkpoint.pth.tar
python -m torch.distributed.launch --nproc_per_node=1 --master_port=11111   eval_linear_robustness.py --evaluate --dataset imagenet_c --data_path data_path \
--arch SLaK_tiny --kernel_size 3 3 3 3 100 --LoRA False --bn True \
--pretrained_weights $ck1 --url $ck2 \
--output_dir ./ --checkpoint_key teacher

