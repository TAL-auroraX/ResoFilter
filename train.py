# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import os, json
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    tokenizer_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess_my(json_data, tokenizer, model_max_length):
    """
        <bos><start_of_turn>user
        Write a hello world program<end_of_turn>
        <start_of_turn>model
        XXXXX
        <end_of_turn><eos>
    """
    instruction = json_data["instruction"]
    input = json_data["input"]
    output = json_data["output"]

    IGNORE_INDEX = -100
    query = instruction + input
    answer = output
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
    assert len(input_ids) == len(labels)
    input_ids += [tokenizer.pad_token_id] * (model_max_length - len(input_ids))
    labels += [IGNORE_INDEX] * (model_max_length - len(labels))
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        # ret = preprocess([self.raw_data[i]], self.tokenizer, self.max_len)
        ret = preprocess_my(self.raw_data[i], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["labels"],
            attention_mask=ret["attention_mask"],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    train_data = []
    if data_args.data_path.endswith(".jsonl"):
        with open(data_args.data_path, "r") as f:
            for line in f:
                train_data.append(json.loads(line))
    elif data_args.data_path.endswith(".json"):
        with open(data_args.data_path, "r") as f:
            train_data = json.load(f)
    else:
        raise ValueError("ERROR: Unknown data format!!!!!!")
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)
    eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    # print(model_args.tokenizer_name_or_path)
    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        quantization_config=None,
        torch_dtype=torch.float16,
        attn_implementation='eager'
    )
    # model.resize_token_embeddings(151936 + 5)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
