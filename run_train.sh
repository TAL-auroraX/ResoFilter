#!/bin/bash
GPUS=0,1,2,3,4,5,6,7
MODEL=google/gemma-2-2b-it
TOKENIZER=${MODEL}
TRAIN_DATA=train_data/mean_diff_p25.jsonl
OUTPUT_DIR=train_output/
CACHE_DIR=${OUTPUT_DIR}
MAX_LENGTH=768
EPOCH=1
DS_CONFIG=zero2.json

mkdir -p ${OUTPUT_DIR}

deepspeed --include localhost:${GPUS} train.py \
    --deepspeed ${DS_CONFIG} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name_or_path ${TOKENIZER} \
    --data_path ${TRAIN_DATA} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCH} \
    --cache_dir ${CACHE_DIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length ${MAX_LENGTH} \
    --gradient_checkpointing False \
    --lazy_preprocess True
