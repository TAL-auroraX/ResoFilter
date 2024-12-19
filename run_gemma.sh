export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export WANDB_DISABLED=True

python run_gemma.py \
    --data_path data/MetaMathQA.json \
    --save_dir_path result/ \
    --model_name_or_path google/gemma-2-2b \
    --max_length 768 \
    --save_step 10000 \
    --if_remove_dump True

