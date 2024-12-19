python reso_filter.py \
    --data_path data/MetaMathQA.json \
    --diff_result_path result/ \
    --save_dir_path train_data/ \
    --target_layer_type "mlp.up_proj.weight" \
    --max_layer 26

