#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name=arxiv-large

output=snap/$name

PYTHONPATH=$PYTHONPATH:./flan_arxiv_src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12321 \
    flan_arxiv_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
	--gradient_accumulation_steps 4 \
        --train Arxiv \
        --valid Arxiv \
        --batch_size 8 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'link,classification' \
        --backbone 'google/flan-t5-large' \
        --output $output ${@:2} \
        --epoch 2 \
	--weight_decay 0 \
        --max_text_length 512 \
        --gen_max_length 64 \
	--lr 0.00008
