

ck1=/home/shiwei/Projects/dino/ckpts/300epochs/dino_vit_small/checkpoint.pth
ck2=/home/shiwei/Projects/dino/ckpts/300epochs/dino_vit_small/linear/checkpoint.pth.tar
path=/home/shiwei/data/
python -m torch.distributed.launch --nproc_per_node=1 --master_port=11112 eval_linear_robustness.py --evaluate --data_path $path --dataset imagenet_c \
--arch vit_small --pretrained_weights $ck1 --url $ck2  --output_dir ./ --checkpoint_key teacher \


