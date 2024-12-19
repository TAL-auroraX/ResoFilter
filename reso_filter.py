from tqdm import tqdm
from copy import deepcopy as dp
import datasets, json, os, argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/MetaMathQA.json')
    parser.add_argument("--diff_result_path", type=str, default='result/')
    parser.add_argument("--save_dir_path", type=str, default='train_data/')
    parser.add_argument("--target_layer_type", type=str, default="mlp.up_proj.weight", help=
                        "model layer type when using transformers as loading method, taking one each time.")
    parser.add_argument("--max_layer", type=int, default=26, help=
                        "max layer index of the model, for gemma-2-2B it's 26, for LlaMA-2 7B it's 32.")
    parser.add_argument("--metric", type=str, default='mean_diff', help=
                        "the metric you choose to analyze the difference. In the run_gemma.py we provided multiple metrics choices to calculate the difference.")
    parser.add_argument("--if_remove_dump", type=bool, default=True)

    args = parser.parse_args()
    return args


def _add_idx(example, _idx):
    """to each example, add an idx for fast locating the origin sample"""
    example['idx'] = _idx
    return example


def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_dataset_to_format_train(_dataset: datasets.Dataset, _file_path):
    with open(_file_path, 'w', encoding='utf-8') as f:
        for row in _dataset:
            _new_dict = {
                'instruction': "",
                'input': row['query'],
                'output': row['response'],
                'idx': row['idx']
            }
            f.write(json.dumps(_new_dict, ensure_ascii=False) + '\n')


def remove_dum(_dataset: datasets.Dataset):
    _new_dataset_idx = []
    content_record = set()
    for _idx, row in enumerate(_dataset):
        content = row['query']
        if content in content_record:
            continue
        content_record.add(content)
        _new_dataset_idx.append(_idx)
    print(f"origin dataset size: {_dataset.num_rows}, after removing size: {len(_new_dataset_idx)}")
    return _dataset.select(_new_dataset_idx)


def main(args):
    math_data_path = args.data_path
    math_dataset = datasets.load_dataset('json', data_files=math_data_path, split='train')
    math_dataset = math_dataset.map(_add_idx, with_indices=True)

    files = os.listdir(args.diff_result_path)
    diff_data = []
    for file_name in tqdm(files, desc="loading difference files..."):
        if args.target_layer_type.split(".")[1] in file_name:
            file_path = os.path.join(args.diff_result_path, file_name)
            _tmp = read_json(file_path)
            diff_data.extend(_tmp)
    print(f"total {len(diff_data)} difference data loaded.")

    top3_layer_name = [f"model.layers.{x}.{args.target_layer_type}" for x in range(args.max_layer-3, args.max_layer)]
    attr_name = args.metric
    all_diffs = {}
    for info in diff_data:
        value = sum([info[name][attr_name] for name in top3_layer_name]) / len(top3_layer_name)
        all_diffs[info['idx']] = value

    sorted_diffs = sorted(all_diffs.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    if args.if_remove_dump:
        nodup_datasets = remove_dum(math_dataset)
    else:
        nodup_datasets = dp(math_dataset)

    if not os.path.exists(args.save_dir_path):
        os.makedirs(args.save_dir_path)

    for percent in [0.25, 0.5, 0.75]:
        percent_num = int(len(nodup_datasets) * percent)
        valid_datasets = nodup_datasets.select([x[0] for x in sorted_diffs])
        final_datasets = valid_datasets.select(list(range(percent_num, len(valid_datasets))))
        df = final_datasets.to_pandas()
        sorted_indices = df['idx'].argsort()
        sorted_datasets = final_datasets.select(sorted_indices)
        file_path = os.path.join(args.save_dir_path, f"{attr_name}_p{int((1-percent)*100)}.jsonl")
        save_dataset_to_format_train(sorted_datasets, file_path)
        print(sorted_datasets)
        print(f"save to {file_path}")
        print("#"*30)


if __name__ == '__main__':
    args = parse_args()
    main(args)



