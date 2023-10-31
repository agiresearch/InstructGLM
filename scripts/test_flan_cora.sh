# Run with $ bash scripts/pretrain_P5_base_beauty.sh 2

#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7

name=arxiv-base

output=snap/$name

PYTHONPATH=$PYTHONPATH:./cora_flan_src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12322 \
    cora_flan_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
	--gradient_accumulation_steps 1 \
        --train Cora \
        --valid Cora \
        --batch_size 50 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'classification' \
        --backbone 'google/flan-t5-base' \
        --output $output ${@:2} \
        --epoch 6 \
	--inference \
	--weight_decay 0 \
        --max_text_length 512 \
        --gen_max_length 64 \
	--lr 0.00008
