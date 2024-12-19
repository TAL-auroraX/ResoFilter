from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from copy import deepcopy as dp
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import torch, datasets, json, math, os, copy, argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/MetaMathQA.json')
    parser.add_argument("--save_dir_path", type=str, default='result/')
    parser.add_argument("--model_name_or_path", type=str, default='gpt2')
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--layer_type", type=str, default="mlp.up_proj.weight", help=
                        "model layer type when using transformers as loading method, taking ',' as split symbol")
    parser.add_argument("--save_step", type=int, default=10000)
    parser.add_argument("--if_remove_dump", type=bool, default=True)
    args = parser.parse_args()
    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # make sure this port is available.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

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

def init_dataset(tokenizer, data_path, max_length, if_remove_dump):
    def tokenize_function(json_data):
        """
            Gemma instruction format:
            <bos><start_of_turn>user
            Write a hello world program<end_of_turn>
            <start_of_turn>model
            XXXXX
            <end_of_turn><eos>
        """
        model_max_length = max_length
        IGNORE_INDEX = -100
        query = json_data["query"]
        answer = json_data["response"]
        # query = json_data["input"]
        # answer = json_data["output"]

        if type(query) == str:
            source_prompt = "<start_of_turn>user\n{}<end_of_turn>\n"
            target_prompt = "<start_of_turn>model\n{}<end_of_turn>"
            source = source_prompt.format(query)
            target = target_prompt.format(answer)
            sentence_ids = tokenizer.encode(source)
            target_ids = tokenizer.encode(target)[1:]
            input_ids = sentence_ids + target_ids + [tokenizer.eos_token_id]
            labels = [IGNORE_INDEX] * len(sentence_ids) + target_ids + [tokenizer.eos_token_id]
        assert len(input_ids) == len(labels)
        input_ids = input_ids[: model_max_length]
        labels = labels[: model_max_length]
        attention_mask = [1] * len(input_ids)
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return tokenized_full_prompt

    math_dataset = datasets.load_dataset('json', data_files=data_path, split='train')
    if if_remove_dump:
        math_dataset = remove_dum(math_dataset)
    # math_dataset = math_dataset.select(range(10))
    tokenized_dataset = math_dataset.map(tokenize_function, num_proc=128)
    return tokenized_dataset


def euclidean_similarity(A, B):
    return torch.norm(A - B)


def cosine_similarity(A, B):
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    return torch.dot(A_flat, B_flat) / (torch.norm(A_flat) * torch.norm(B_flat))


def pearson_correlation(A, B):
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    A_mean = torch.mean(A_flat)
    B_mean = torch.mean(B_flat)

    numerator = torch.sum((A_flat - A_mean) * (B_flat - B_mean))
    denominator = torch.sqrt(torch.sum((A_flat - A_mean) ** 2) * torch.sum((B_flat - B_mean) ** 2))

    return numerator / denominator


def ssim_similarity(A, B, window_size=5, size_average=True):
    import torch.nn.functional as F
    C1 = 1e-6 ** 2
    C2 = 3e-6 ** 2

    mu1 = F.avg_pool2d(A, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(B, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(A * A, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(B * B, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(A * B, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def frobenius_similarity(A, B):
    return torch.norm(A - B, p='fro')


def manhattan_similarity(A, B):
    return torch.sum(torch.abs(A - B))

def compute_param_diff_llama(params1, params2, valid_layer, device):
    """
        The main function to calculate the difference between two models weights.
        For the efficiency, here we only calculate the mean difference of the weights.
        But we provide all the measures we used or trial in the paper as the following annotated code,
        so feel free to calculate the std, max, min, etc. as your interest.
    """
    differences = {}

    def batch_quantile(tensor, q, batch_size=1000000):
        results = []
        for i in range(0, tensor.numel(), batch_size):
            batch = tensor.flatten()[i:i+batch_size]
            results.append(torch.quantile(batch, q))
        return torch.stack(results).mean(dim=0)

    bounds = [1e-6, 5e-6, 9e-6, 1e-5]

    for name in params1.keys():
        if name in params2 and any([layer in name for layer in valid_layer]):
            with torch.no_grad():
                p1, p2 = params1[name].float().to(device), params2[name].float().to(device)
                diff = p1 - p2
                # statistical measures
                mean_diff = diff.abs().mean().item()
                # std_diff = diff.std().item()
                # max_diff = diff.max().item()
                # min_diff = diff.min().item()
                # std_abs_diff = diff.abs().std().item()
                #
                # euc_sim = euclidean_similarity(p1, p2).item()
                # cos_sim = cosine_similarity(p1, p2).item()
                # pearson_sim = pearson_correlation(p1, p2).item()
                # frob_sim = frobenius_similarity(p1, p2).item()
                # man_sim = manhattan_similarity(p1, p2).item()
                # # (batch_size, channels, height, width)
                # A_4d = p1.unsqueeze(0).unsqueeze(0)
                # B_4d = p2.unsqueeze(0).unsqueeze(0)
                # ssim_sim = ssim_similarity(A_4d, B_4d).item()
                # # calculate percentiles
                # percentiles = torch.tensor([0.1, 0.25, 0.9, 0.95, 0.99], dtype=torch.float32, device=device)
                # percentile_values = batch_quantile(diff.abs(), percentiles).tolist()

                differences[name] = {
                    "mean_diff": mean_diff,
                    # "std_diff": std_diff,
                    # "max_diff": max_diff,
                    # "min_diff": min_diff,
                    # "std_abs_diff": std_abs_diff,
                    # "10th": percentile_values[0],
                    # "25th": percentile_values[1],
                    # "90th": percentile_values[2],
                    # "95th": percentile_values[3],
                    # "99th": percentile_values[4],
                    # "euc": euc_sim,
                    # "cos": cos_sim,
                    # "pearson": pearson_sim,
                    # "frob": frob_sim,
                    # "man": man_sim,
                    # "ssim": ssim_sim
                }
                # for bound in bounds:
                #     total_change_count = (diff.abs() <= bound).sum().item()
                #     positive_change_count = (diff > bound).sum().item()
                #     negative_change_count = (diff < -bound).sum().item()
                #     differences[name][f"no_change_{bound}"] = total_change_count
                #     differences[name][f"pos_change_{bound}"] = positive_change_count
                #     differences[name][f"neg_change_{bound}"] = negative_change_count

    return differences

def reformat_data(data, device):
    return {
        "input_ids": torch.tensor(data["input_ids"]).unsqueeze(0).to(device),
        "attention_mask": torch.tensor(data["attention_mask"]).unsqueeze(0).to(device),
        "labels": torch.tensor(data["labels"]).unsqueeze(0).to(device)
    }

def train(rank, world_size, tokenized_dataset, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    valid_layer = args.layer_type.split(",")
    model = create_model(args.model_name_or_path, device)
    original_params = dp(model.state_dict())
    data_record = []
    # choose datasets
    data_range = len(tokenized_dataset) // world_size
    slice_dataset = tokenized_dataset.select(range(rank * data_range, (rank + 1) * data_range))
    if rank == world_size - 1:
        slice_dataset = tokenized_dataset.select(range(rank * data_range, len(tokenized_dataset)))
    file_idx = 0
    for idx, sample in tqdm(enumerate(slice_dataset), total=len(slice_dataset)):
        model.load_state_dict(original_params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0)
        optimizer.zero_grad()
        model.zero_grad()
        inputs = reformat_data(sample, device)
        outputs = model(**inputs)
        loss = outputs.loss
        # calculate ppl
        loss_value = loss.item()
        total_loss = loss_value * inputs["input_ids"].size(1)
        total_tokens = inputs["input_ids"].size(1)
        ppl = math.exp(total_loss / total_tokens)
        # 计算diff
        loss.backward()
        optimizer.step()
        param_diff = compute_param_diff_llama(original_params, model.state_dict(), valid_layer, device)

        # 保存相关信息
        param_diff["idx"] = rank * data_range + idx
        param_diff["loss"] = loss.item()
        param_diff["ppl"] = ppl
        if "id" in sample:
            param_diff["id"] = sample["id"]
        data_record.append(param_diff)
        if (idx != 0 and idx % args.save_step == 0) or idx == len(slice_dataset) - 1:
            for layer_name in valid_layer:
                all_data = []
                for record in data_record:
                    same_layer_data = {}
                    for key in record.keys():
                        if layer_name in key:
                            same_layer_data.update({key: record[key]})
                    same_layer_data["idx"] = record["idx"]
                    same_layer_data["loss"] = record["loss"]
                    same_layer_data["ppl"] = record["ppl"]
                    if "id" in record:
                        same_layer_data["id"] = record["id"]
                    all_data.append(same_layer_data)
                file_middle_name = layer_name.split(".")[1]
                save_path = os.path.join(args.save_dir_path, f"diff_{file_middle_name}_device{rank}_part{file_idx}.json")
                f = open(save_path, "w")
                json.dump(all_data, f, indent=4)
                f.close()
            file_idx += 1
            data_record = []
    cleanup()

def create_model(model_name_or_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device, attn_implementation='eager')
    return model

def main():
    args = parse_args()
    world_size = os.environ['CUDA_VISIBLE_DEVICES'].count(',') + 1
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    tokenized_dataset = init_dataset(tokenizer, args.data_path, args.max_length, args.if_remove_dump)
    if not os.path.exists(args.save_dir_path):
        os.makedirs(args.save_dir_path)

    mp.spawn(train,
             args=(world_size, tokenized_dataset, args),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()

