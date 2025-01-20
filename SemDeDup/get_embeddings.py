import os
import pdb
import sys
import random
import logging
import numpy as np
from pathlib import Path
from functools import partial
from typing import Optional, Union
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    set_seed,
)

from compute_pretrained_embeddings import get_embeddings


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    padding_side: str = field(
        default="left",
        metadata={
            "help": "The side on which the model should have padding for the input"
        },
    )
    truncation_side: str = field(
        default="left",
        metadata={
            "help": "The side on which the model should have truncating for the input"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. It may cause errors"
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention in the model training"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a json/jsonl file)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set.",
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )

    def __post_init__(self):
        assert self.eval_file is not None, "eval_file is required."

        extension = self.eval_file.split(".")[-1]
        assert extension in [
            "json",
            "jsonl",
        ], "`eval_file` should be a json or a jsonl file."


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    seed: int = field(default=42, metadata={"help": "Random seed for reproduciblity."})
    data_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed to be used with data samplers. If not set, random generators for data sampling will use the same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model seed."
        },
    )
    report_to: str = field(
        default="wandb",
        metadata={
            "help": 'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        },
    )
    output_dir: str = field(
        default="./output",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory."
        },
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={
            "help": "When performing evaluation and generating predictions, only returns the loss."
        },
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": "Log the training loss and learning rate every logging_steps steps."
        },
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to group samples of roughly the same length together when batching."
        },
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={
            "help": "Column name with precomputed lengths to use when grouping by length."
        },
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )


def prepare_model_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "padding_side": model_args.padding_side,
        "truncation_side": model_args.truncation_side,
    }
    if model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        )

    # Adjust tokenizer: no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    assert isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ), "Only llama Model is supported yet."
    num_added_tokens = tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<unk>",
        }
    )
    assert (
        num_added_tokens == 0
    ), "LlamaTokenizer should not add any token (when pad_token is set as unk_token)."

    # load model
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        use_flash_attention_2=model_args.use_flash_attn,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        token=model_args.token,
        device_map="auto",
    )

    return model, tokenizer


def text2token(example, index, tokenizer, max_seq_length):
    # todo: modify "predict_instruction" to another field based on your deduplication target
    example_text = example["predict_instruction"]

    example_text = example_text.strip() + "\n"
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )

    return {
        "input_ids": tokenized_example.input_ids.flatten(),
        "index": index,
    }


def prepare_data(training_args, data_args, model_args, tokenizer):
    raw_dataset = datasets.load_dataset(
        "json",
        data_files={"eval": data_args.eval_file},
        cache_dir=model_args.cache_dir,
    )["eval"]

    # set the format to pytorch
    raw_dataset.set_format(type="pt")

    # select the max_eval_samples samples, if necessary
    if (
        data_args.max_eval_samples is not None
        and len(raw_dataset) > data_args.max_eval_samples
    ):
        raw_dataset = raw_dataset.select(range(data_args.max_eval_samples))

    # Tokenize the dataset
    encode_function = partial(
        text2token,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )
    eval_dataset = raw_dataset.map(
        encode_function,
        with_indices=True,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        # remove_columns=[
        #     name
        #     for name in raw_dataset.column_names
        #     if name not in ["input_ids", "attention_mask"]
        # ],
        desc="Tokenizing and reformatting instruction data",
    )

    # # add index to the dataset
    # add_index_func = lambda x, i: {"index": i}
    # eval_dataset = raw_dataset.map(
    #     add_index_func,
    #     with_indices=True,
    #     batched=False,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    #     desc="Adding index to the dataset",
    # )

    # Prepare group by length
    if training_args.group_by_length:
        eval_dataset = eval_dataset.map(
            lambda x: {"length": int(len(x["input_ids"]))},
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Computing dataset lengths",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

    return {"eval_dataset": eval_dataset}


@dataclass
class DataCollator:
    """
    Modified from transformers.DataCollatorForSeq2Seq. All rights are reserved to the original authors.

    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, transformers.tokenization_utils_base.PaddingStrategy] = (
        "longest"
    )
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, samples, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        tokenized_inputs = [s["input_ids"] for s in samples]
        indices = [int(s["index"]) for s in samples]

        padded_inputs = self.tokenizer.pad(
            {"input_ids": tokenized_inputs},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return {
            "data": padded_inputs.data,
            "path": [
                "index " + str(int(s["index"])) + ": " + s["prompt"] for s in samples
            ],
            "index": indices,
        }


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Prepare output directory
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(output_dir)} is created.")

    # Prepare the logger file
    file_handler = logging.FileHandler(output_dir / "run.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        f"{'-'*25 + 'Configurations' + '-'*25}\n"
        f"Model args: {model_args}\n"
        f"Data args: {data_args}\n"
        f"Training args: {training_args}\n"
        f"{'-'*25 + 'Configurations Complete' + '-'*25}\n"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_tokenizer(model_args)

    # Prepare the datasets.
    dataset = prepare_data(training_args, data_args, model_args, tokenizer)[
        "eval_dataset"
    ]
    data_collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=data_args.preprocessing_num_workers,
        # pin_memory=True,
    )

    # load dataloader
    dataset_size = len(dataset)
    emb_size = model.config.hidden_size

    emd_fpath = output_dir / "embeddings.npy"
    paths_fpath = output_dir / "paths.npy"
    # emd_fpath.write_bytes(np.zeros((dataset_size, emb_size), dtype="float32").tobytes())
    # paths_fpath.write_text("\n".join(dataset["index"]))
    np.zeros((dataset_size, emb_size), dtype="float32").tofile(emd_fpath)
    np.empty(dataset_size, dtype="U10000").tofile(paths_fpath)

    emd_memmap = np.memmap(
        emd_fpath,
        dtype="float32",
        mode="w+",
        shape=(dataset_size, emb_size),
    )
    paths_memmap = np.memmap(
        paths_fpath,
        dtype="U10000",
        mode="w+",
        shape=(dataset_size,),
    )
    get_embeddings(model, dataloader, emd_memmap, paths_memmap)


if __name__ == "__main__":
    main()

    print("Hello World!")
