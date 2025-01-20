# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import getpass
from datetime import datetime
import random
import numpy as np
import torch
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List
from collections.abc import Mapping
import re
import sys
import argparse
MAX_INT = sys.maxsize
MAX_LENGTH = int(1024)  # Hardcoded max length to avoid infinite loop

def deepcopy_fsdp_models(src, tgt):
    """Given two models src and tgt, copy every parameter from the src to the tgt model."""
    with torch.no_grad():
        src_params = { k: v for k,v in src.named_parameters() }
        tgt_params = { k: v for k,v in tgt.named_parameters() }

        for k in tgt_params:
            if k in src_params:
                tgt_params[k].data.copy_(src_params[k].data.detach())
            else:
                rank0_print(f"{k} not found")


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def on_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False, token_level: bool = False):
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If true, return the token-level log probabilities (do not aggregate across tokens)

    Returns:
        The relevant log probabilities. Of shape (batch_size,) by default and shape (batch size, sequence length) if token_level.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    distribution_logps = logits.log_softmax(-1)

    per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if token_level: 
        return (per_token_logps * loss_mask)
    elif average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    return variance


def rowwise_product(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the row-wise product over all the elements that have not been masked out.

    Args:
        mat: tensor of shape (batch_size, sequence length)
        mask: tensor of shape (batch_size, sequence length) 

    Returns:
        Matrix of batch size. 
    """
    mat = mat.clone()
    indices = (mask == 0).long().nonzero()
    mat[indices[:,0], indices[:,1]] = 1
    return mat.prod(dim=1)


def entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits.
    
    Args:
        logits: tensor of shape (batch_size, sequence length, vocab)
        mask: tensor of shape (batch_size, sequence length)
    
    Returns:
        The average tokenwise entropy across all non-masked tokens (of shape (1,)).
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = masked_mean(torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1), mask)
    return entropy


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    device = torch.device('cuda', rank)
    all_values = [torch.empty_like(values).to(device) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)



def detect_abnormal_value(input_tensor: torch.Tensor):
    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        return True
    else:
        return False



# newly-add

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=False, help="llama2-70b-chat")
    parser.add_argument("--saved_policy", default=None, type=str, required=False, help="model path")
    parser.add_argument("--tokenizer_dir", default=None, type=str, required=False, help="tokenizer path")
    parser.add_argument("--input_file", default=None, type=str, required=False, help="input data to LLM")
    parser.add_argument("--output_file", default=None, type=str, required=False, help="save your result")
    parser.add_argument("--data_source", default="hh", type=str, required=False, help="indicate your dataset source, e.g., hh or mmlu")
    parser.add_argument("--specify_your_gpus", default="0,1,2,3,4,5,6,7", type=str, required=False, help="your available gpus")
    parser.add_argument("--nshots", default=None, type=int, required=False, help="few shot input")
    parser.add_argument('--model.max_new_length', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument("--num_try", type=int, default=-1) #only required for uncertain method
    parser.add_argument("--chunk", type=int, default=-1) # parallel inference with multiple gpus
    parser.add_argument('--stop', type=str, nargs='+', default=[],
                        help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='all')
    parser.add_argument('--torch_type', type=str, default='all')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--max_num_batched_tokens', type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--benchmark', action="store_true", help="additionally run benchmark")
    parser.add_argument('--use_meta_tensor', action="store_true", help="use the meta tensors to initialize model")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save_mp_checkpoint_path", required=False, default=None, type=str, help="save-path to store the new model checkpoint")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--pad_token_id", type=int, default=None, help="pad token id")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
    parser.add_argument("--replace_method", required=False, default='auto', type=str, help="replace method['', 'auto']")
    parser.add_argument("--do_sample", action='store_true', help="sampling for generation")
    parser.add_argument("--query_suffix", type=str, default=None, help="the suffix for harmful queries")
    parser.add_argument("--logprobs", type=int, default=None, help="Number of log probabilities to return per output token.")

    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code from huggingface.")


    # args = parser.parse_args()

    return parser


def print_latency(latency_set, title=""):
    # 10 warmup queries
    latency_set = latency_set[10:]
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print("====== latency stats {0} ======", title)
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    # for i in range(n-1):
    for i in range(n):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n) * batch_size
    if last_start < len(data_list):
        last_end = MAX_INT
        batch_data.append(data_list[last_start:last_end])
    return batch_data



def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def next_option_letter(letter):
    if letter == 'Z':
        return 'A'
    else:
        return chr(ord(letter) + 1)


def delete_extra_zero(n):
    '''Delete the extra 0 after the decimal point'''
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n


from transformers import AutoTokenizer, pipeline
from packaging.version import Version
from datasets import Dataset
from multiprocessing import Pool
import time
from typing import List, Dict, Any
from tqdm import tqdm
import pdb

# Create a class to do batch inference.
class LLMPredictor:
    def __init__(
            self,
            model_path,
            tokenizer_path,
            prompt_key="prompt",
            generation_key="generation",
            max_tokens=2048,
            max_model_len=4096,
            num_generations=1,
            temperature=0.0,
            top_p=1.0,
            stop_tokens=None,
            stop_token_ids=None,
            tensor_parallel_size=1,
            enable_prefix_caching=False,
            swap_space=16,
    ):
        from vllm import LLM, SamplingParams

        seed = int(time.time() * 1e6) % int(1e9)
        # Create an LLM.
        self.prompt_key = prompt_key
        self.generation_key = generation_key
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True,
            swap_space=swap_space,
            gpu_memory_utilization=0.95,
            seed=seed,
            dtype=torch.float32
        )
        print("============loaded model===================")
        self.sampling_params = SamplingParams(
            n=num_generations,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch[self.prompt_key], self.sampling_params)
        generated_text: List[str] = []
        for output in outputs:
            generated_text.append([o.text for o in output.outputs])
        return {**batch, self.generation_key: generated_text}


def run_vllm_inference_distributed(
        ds,
        **kwargs,
):
    import ray
    import ray.data
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)

    # Guarentee the compute resources is available
    if torch.cuda.device_count() < tensor_parallel_size:
        raise MemoryError(
            "Insufficient GPUs: tensor_parallel_size ({}) < available gpus ({})".format(
                tensor_parallel_size, torch.cuda.device_count()
            )
        )

    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = torch.cuda.device_count() // tensor_parallel_size
    print("Launch {} instances for vllm inference.".format(num_instances))

    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size, strategy="STRICT_PACK"
        )
        return dict(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True
            )
        )

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    batch_size = min(ds.count() // num_instances + 1, 10000)
    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=batch_size,
        fn_constructor_kwargs=kwargs,
        **resources_kwarg,
    )

    return ds


def process_data_on_device(device, sub_dataset, score_model_path, score_tokenizer):
    score_pipe = pipeline(
        "sentiment-analysis",
        model=score_model_path,
        device=device,
        tokenizer=score_tokenizer,
        # model_kwargs={"torch_dtype": torch.bfloat16},
        model_key={"torch_dtype": torch.float32},
        truncation=True
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1,
    }

    def get_reward(test_texts):
        pipe_outputs = score_pipe(test_texts, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards

    all_data = []
    for line_data in tqdm(sub_dataset):
        score = get_reward(line_data["predict_instruction"])
        line_data["model_score"] = score
        # line_data["score_model"] = score_model_path
        all_data.append(line_data)

    return all_data


def process_wrapper(args):
    return process_data_on_device(*args)


def generate_score(
        dataset,
        model_path="/path/to/difficulty_score_model",
        tokenizer_path="/path/to/difficulty_score_model",
        args=None

):
    num_gpus = torch.cuda.device_count()

    # sub_datasets = [dataset.shard(num_shards=num_gpus, index=i) for i in range(num_gpus)]
    score_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # with Pool(num_gpus) as p:
    #     results = p.map(process_wrapper, [(i, sub_datasets[i], model_path, score_tokenizer) for i in range(num_gpus)])
    # all_data = [item for sublist in results for item in sublist]

    all_data = process_data_on_device(args.specify_your_gpus, dataset, model_path, score_tokenizer)
    pdb.set_trace()

    final_dataset = Dataset.from_list(all_data)



STRIP_STRS = [
    ":",
    ".",
    "/",
    ",",
    "#",
    "?",
    "$",
    '"',
    "'",
    # "ки" is the delimeter for Math-Shepherd
    "к",
    "и",
    # LaTeX
    "\\(",
    "\\)",
    "\\[",
    "\\]",
]
NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS
NO_PRECEDING_PUNCS = ["!", ")", "]", "}", "\\\\"] + STRIP_STRS
# Answer prefixes
PRM800K_ANS_PRRFIX = "# Answer"
GSM8K_ANS_PREFIX = "####"


def extract_boxed(resp: str) -> str:
    ans = resp.split("oxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()  # todo: ?
    return a


def clean_preceding(
    s: str,  # The input string.
) -> str:  # The cleaned string with preceding punctuation marks removed.
    """Removes preceding punctuation marks from a string."""
    s = str(s).strip()
    while s != "" and s[0] in NO_PRECEDING_PUNCS:
        s = s[1:].strip()
    return s


def clean_trailing(
    s: str,  # The input string.
) -> str:  # The cleaned string with trailing punctuation marks removed.
    """Removes trailing punctuation marks from a string."""
    s = str(s).strip()
    while s != "" and s[-1] in NO_TRAILING_STRS:
        s = s[:-1].strip()
    return s


def clean(ans: str) -> str:
    """Clean the extracted answer."""

    ans = ans.strip()
    ans = clean_preceding(ans)
    ans = clean_trailing(ans)

    return ans

# https://github.com/hkust-nlp/dart-math/blob/23666f98f53b9e5289e0bca5492b59a6aca2df42/dart_math/eval.py#L192


def check_safetensor_exists(directory, suffix='.safetensor'):
    # List all files in the specified directory
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            # Check if the file ends with .safetensor
            if filename.endswith(suffix):
                return True
    return False


# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer



def extract_numbers(input_string):
    # Use regular expression to find all numbers including decimals
    numbers = re.findall(r'\d+\.?\d*', input_string)
    # Convert found strings to float
    res = [float(num) for num in numbers]
    if len(res) >= 1:
        return res[0]
    else:
        return input_string



def extract_hash(string):
    ans_pattern = "####(.*)"
    ans = re.findall(ans_pattern, string)
    if len(ans)>=1:
        ans = ans[-1].strip()
    else:
        ans = string
    return ans



def change_boxed_to_hash(string, ans):
    # # Define a pattern to match LaTeX boxed expressions
    # pattern = r'(\w+\s+is:\n+\\\[\n\\boxed{(.*?)}\n\\\])'
    #
    # # Function to format the matched groups
    # def replace(match):
    #     question = match.group(1).split('is:')[0]  # Get the question part
    #     answer = match.group(2)  # Get the boxed answer
    #     return f"{question}is: #### {answer}"
    #
    # # Replace all occurrences in the input string
    # return re.sub(pattern, replace, string)

    prior = string.split("boxed")[0].strip()
    symbols = ["\\", "[", "]", "("]
    while prior[-1] in ["\\", "[", "]", "("]:
        for s in symbols:
            prior = prior.rstrip(s).strip()
    return prior.strip() + " #### " + ans
