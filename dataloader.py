# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the functions for loading data.
Each function of the form get_{dataset_name} (e.g., get_hh_query, get_mmlu, etc.) will return a dict of Example objects, indexed by the prompt for the text.

Each Example object will contain
- the prompt (formatted with config.human_prefix, config.assistant_prefix)
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores for the generations
- for binary feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for unary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
- the explore_dataset name
- the unformatted prompt (needed for alpaca)
"""
import copy
import json
import pdb
import sys
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
from tqdm.contrib import tzip

import re
import os
import random
from omegaconf import OmegaConf, DictConfig
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from utils.fastchat_conversation_template import get_conv_template
from utils.common_prompt import wrap_query_to_prompt
from utils.common_funcs import rank0_print, on_rank0, delete_dict


@dataclass
class Example:
    """
    Class for an example in a preference or SFT explore_dataset. If you want each prompt to be uniquely associated with an Example instance, save it in a dict.
    """
    prompt: str = ''                                            # prompt for the generated texts
    generations: List[str] = field(default_factory=list)        # list of generations
    sft_index: int = -1                                         # which response in generations should be generated for SFT
    scores: List[float] = field(default_factory=list)           # score for each generation
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # for binary feedback data:: indices in responses, where i > j in pair (i,j) is a preference
    desirable: List[bool] = field(default_factory=list)         # for unary feedback data: whether the generation at the corresponding index in self.generations is desirable
    truncation_mode: str = 'keep_end'                           # if truncation needed, keep the beginning (keep_start) or end (keep_end) (only override default for SHP)
    dataset_name: str = ''
    original_prompt: str = ''                                   # the unformatted prompt (needed to recover instruction for AlpacaEval)

    def num_generations(self):
        return len(self.generations)

    def remove_extra_spaces(self):
        """
        Remove double spaces in certain datasets, like Anthropic HH, to standardize spacing.
        """
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        self.prompt = clean(self.prompt)
        self.generations = list(map(clean, self.generations))


class Dataset:
    """
    A collection of Example instances, indexed by prompt.
    """
    def __init__(self, name):
        self.name = name
        self.data = defaultdict(Example)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("key must be a string")
        
        if not isinstance(value, Example):
            raise ValueError("value must be a Example")
        
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)

def prepare_data(task_name: str, split: str, start_idx: int, end_idx: int, prev_fw: str, config: DictConfig, prompt_key="prompt"):
    if split == "train":
        dataset = json.load(open(config.train_file))
        if start_idx != -1 or end_idx != -1:
            dataset = dataset[start_idx: end_idx]
    else:

        dataset = json.load(open(config.test_file))
        if start_idx != -1 or end_idx != -1:
            dataset = dataset[start_idx: end_idx]
        print(f"{task_name}: loading {config.test_file} from {start_idx} to {end_idx} ...")

        # remove the data we have generated
        if os.path.exists(prev_fw):
            with open(prev_fw, 'r', encoding="utf-8") as fr:
                print(f"{prev_fw} has been there, we continue explore_dataset based on it!")
                previous_dataset = [json.loads(item)['prompt'] for item in fr]
                new_dataset = []
                for item in dataset:
                    if item[prompt_key] not in previous_dataset:
                        new_dataset.append(item)
                dataset = new_dataset
            print(f"We have generated {len(previous_dataset)}, we now need to process {len(dataset)} samples ....")

    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing {}'.format(task_name))

    return dataset


# ---- Step-1 Warm-up Step-2 Feedback -----
def get_warmup_data(split: str, start_idx: int, end_idx: int, prev_fw: str, config: DictConfig) -> Dataset:
    """
    Load the instruction prompts from local files and convert it into to a Dataset.
    For this explore_dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - start_idx: index of the first example in the dataset
        - end_idx: index of the last example in the dataset
        - prev_fw: files saving the previously predicted instructions
        - config: args for processing the dataset

    Returns:
        A Dataset instance.
    """
    # read_fn = sys._getframe().f_code.co_name
    task_name = config.exp_name + "_warmup"
    rank0_print(f'Loading {task_name} ({split} split) ...')
    data = Dataset(task_name)

    dataset = prepare_data(task_name, split, start_idx, end_idx, prev_fw, config)

    for item in dataset:
        prompt, target = item["prompt"], item["target"]
        data[prompt].prompt = prompt
        data[prompt].generations.append(target)
        data[prompt].sft_index = 0

        data[prompt].dataset_name = task_name
        data[prompt].remove_extra_spaces()

    return data


# ---- Step-3 Iterative Optimization -----
def get_dpo_data(split: str, start_idx: int, end_idx: int, prev_fw: str, config: DictConfig) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless explore_dataset from Huggingface and convert it into to a Dataset.
    For this explore_dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - start_idx: index of the first example in the dataset
        - end_idx: index of the last example in the dataset
        - prev_fw: files saving the previously predicted instructions
        - config: args for processing the dataset

    Returns:
        A Dataset instance.
    """
    task_name = config.exp_name + "_failure_inducing_dpo_optim"
    rank0_print(f'Loading {task_name} ({split} split) ...')
    data = Dataset(task_name)

    dataset = prepare_data(task_name, split, start_idx, end_idx, prev_fw, config)


    if split == "train":
        for item in dataset:
            prompt, chosen, rejected = item["prompt"], item["positive"], item["negative"]
            # in safety red-teaming, positive means causing harmfulness; negative means only eliciting safety response
            # in honesty calibration, positive means causing target model uncertainty; negative means target model's low entropy
            # in math, positive means causing target model produce mistakes; negative means target model correctly answering
            # chosen = " " + chosen
            # rejected = " " + rejected
            responses = [chosen, rejected]
            i, j = data[prompt].num_generations(), data[prompt].num_generations() + 1
            data[prompt].prompt = prompt
            data[prompt].generations.extend(responses)
            data[prompt].pairs.append((i, j))
            data[prompt].sft_index = 0

            data[prompt].dataset_name = task_name
            data[prompt].remove_extra_spaces()
    else:
        for item in dataset:
            prompt = item["prompt"]
            chosen = item["positive"] if "positive" in item else item["target"]
            # chosen = " " + chosen
            rejected = ""
            responses = [chosen, rejected]
            i, j = data[prompt].num_generations(), data[prompt].num_generations() + 1
            data[prompt].prompt = prompt
            data[prompt].generations.extend(responses)
            data[prompt].pairs.append((i, j))
            data[prompt].sft_index = 0

    return data


# alternative, we can use KTO when we don't have balanced positive and negative data
def get_kto_data(split: str, start_idx: int, end_idx: int, prev_fw: str, config: DictConfig) -> Dataset:
    """
    Args:
        - split: one of 'test', 'train'
        - start_idx: index of the first example in the dataset
        - end_idx: index of the last example in the dataset
        - prev_fw: files saving the previously predicted instructions
        - config: args for processing the dataset

    Returns:
        A Dataset instance.
    """

    task_name = config.exp_name + "_failure_inducing_kto_optim"
    rank0_print(f'Loading {task_name} ({split} split) ...')
    data = Dataset(task_name)

    dataset = prepare_data(task_name, split, start_idx, end_idx, prev_fw, config)

    for item in dataset:
        prompt, chosen_or_rejected, target = item["prompt"], item["label"], item["instruction"]  # item["label"] means positive instruction or negative instruction
        data[prompt].generations.append(target)
        data[prompt].desirable.append(chosen_or_rejected)
        data[prompt].sft_index = 0
        data[prompt].dataset_name = task_name
        data[prompt].remove_extra_spaces()

    return data


def get_ppo_data(split: str, start_idx: int, end_idx: int, prev_fw: str, config: DictConfig) -> Dataset:
    """
    Args:
        - split: one of 'test', 'train'
        - start_idx: index of the first example in the dataset
        - end_idx: index of the last example in the dataset
        - prev_fw: files saving the previously predicted instructions
        - config: args for processing the dataset

    Returns:
        A Dataset instance.
    """
    task_name = config.exp_name + "_failure_inducing_ppo_optim"
    rank0_print(f'Loading {task_name} ({split} split) ...')
    data = Dataset(task_name)

    dataset = prepare_data(task_name, split, start_idx, end_idx, prev_fw, config)

    for item in dataset:

        if split == "train":
            chosen_or_rejected = item["score"]
        else:
            chosen_or_rejected = 0

        prompt, target = item["question_prompt"], item["question"]
        # target = " " + target
        data[prompt].generations.append(target)
        data[prompt].desirable.append(chosen_or_rejected)
        data[prompt].sft_index = 0
        data[prompt].dataset_name = task_name
        data[prompt].remove_extra_spaces()

    return data


def get_explored_data(split: str, start_idx: int, end_idx: int, prev_fw: str, config: DictConfig) -> Dataset:
    """
    In this work, we adopt supervised fine-tuning on the generated instructions and their responses on the target models.
    Args:
        - split: one of 'test', 'train'
        - start_idx: index of the first example in the dataset
        - end_idx: index of the last example in the dataset
        - prev_fw: files saving the previously predicted instructions
        - config: args for processing the dataset

    Returns:
        A Dataset instance.
    """
    task_name = config.exp_name + "_failure_data_sft"
    rank0_print(f'Loading {task_name} ({split} split) ...')
    data = Dataset(task_name)

    if config.prompt_key: prompt_key = config.prompt_key
    else: prompt_key = "prompt"

    if config.target_key: target_key = config.target_key
    else: target_key = "answer"

    dataset = prepare_data(task_name, split, start_idx, end_idx, prev_fw, config, prompt_key)

    from utils.common_prompt import gsm_nshot_chats

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)


    for item in dataset:

        if "gsm" in task_name:
            prompt = gsm_nshot_chats(n=0, question=item[prompt_key], config=config)
            # todo: we use apply_chat_template because we use llama3-8b-instruct for math task
            prompt = tokenizer.apply_chat_template(
                conversation=prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = item[prompt_key]  # todo: customize your prompt as needed

        target = item[target_key]
        data[prompt].prompt = prompt
        data[prompt].generations.append(target)
        data[prompt].sft_index = 0
        data[prompt].dataset_name = task_name
        data[prompt].remove_extra_spaces()

    return data



class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed, since the batcch elements will be different depending
    on whether you're doing SFT, aligning with a pairwise loss like DPO, or alignment with a unary loss like KTO. 
    """
    def __init__(self, 
                 dataset_names: List[str],      # e.g., ['shp', 'oasst']; should have  get_{name} method in this file
                 tokenizer,                     # Huggingface tokenizer object
                 split: str = 'train',
                 batch_size: int = 1,
                 max_length: int = 512,         # max length of prompt + response
                 max_prompt_length: int = 128,  # max length of prompt alone
                 max_prompt_count: int = None,
                 n_epochs: Optional[int] = None,
                 n_examples: Optional[int] = None,
                 human_prefix: str = '\n<|user|>\n',            # marks start of human's turn
                 human_suffix: str = '',                        # marks end of human's turn
                 assistant_prefix: str = '\n<|assistant|>\n',   # marks start of assistant's turn
                 assistant_suffix: str = '',                    # marks end of assistant's turn
                 seed:int = 0,
                 start_idx:int=-1,
                 end_idx:int=-1,
                 exp_name:str='',
                 prev_fw:str='',
                 config:DictConfig = None,
                 **kwargs):
        
        torch.manual_seed(seed)
        random.seed(seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.kwargs = kwargs

        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples
        
        self.full_data = {}
        for name in dataset_names:
            dataset = globals()[f"get_{name}"](split, start_idx=start_idx, end_idx=end_idx, prev_fw=prev_fw, config=config)
            self.full_data.update(dataset.data)
        print(f"size of unique prompts (split: {split}): ", len(self.full_data.keys()))

        if "test" in split and "full" in self.n_examples:
            self.n_examples = len(self.full_data.keys())
            print(f"load {self.n_examples} samples")


    def collate(self, batch: Dict[str, List]) -> Dict:
        """
        Takes a list of examples (dicts, where values are lists of ints [tokens] or strings [the original texts]) and returns a batch of examples,
        PyTorch tensors padded to the maximum length. Strings are passed through.
        """
        if self.tokenizer.pad_token_id is None:
            raise Exception("tokenizer's pad_token_id is not specified")
        
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith('_input_ids'):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        return padded_batch

    def tokenize_batch_element(self, prompt: str, generation: str, truncation_mode: str, prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the concatenation of the two on all relevant elements
            (e.g., tokens, attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the
            concatenated elements will have keys starting with '{prefix}_combined_'.
        """
        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            print("excel the max_length, current length is ", len(prompt_token_ids) + len(generation_token_ids))
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]

        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = self.tokenizer.decode(generation_token_ids, skip_special_tokens=True) + ' ' + self.tokenizer.eos_token

        batch_element = {'prompt_text': prompt, f'{prefix}_text': generation}

        for k,v in self.tokenizer(prompt).items():
            # k: input_ids, attention_mask
            batch_element[f'prompt_{k}'] = v

        for k,v in self.tokenizer(generation).items():
            batch_element[f'{prefix}_{k}'] = v

        # combine the prompt and generation belonging to the same example
        batch_element.update(self.combine_prompt_and_generation(batch_element, batch_element, prefix=prefix))
  
        return batch_element

    def combine_prompt_and_generation(self, prompt_dict: Dict, generation_dict: Dict, prefix: str='target') -> Dict:
        """
        Tokenize the concatenated prompt and generation. 
        
        Note that you cannot just concatenate the input ids, attention mask, etc. after the fact -- as done 
        in the DPO repo -- because of subtle differences. For example, the ID for 'Well' corresponds to no 
        space ('Well') when at the start of a text but a space ('\n Well) when succeeding a newline. Therefore
        we could not get the correct token ID for '\nWell' by first tokenizing '\n' then 'Well' then concatenating
        the resulting tokens together.

        self.tokenizer("Well"): 'input_ids': [1, 5674]
        self.tokenizer("\n"): 'input_ids': [1, 29871, 13]
        self.tokenizer("\nWell"): 'input_ids': [1, 29871, 13, 11284]
        self.tokenizer("\n Well"): 'input_ids': [1, 29871, 13, 5674]

        self.tokenizer.decode([1, 29871, 13] + [5674]): '<s> \n Well' --> an additional empty space

        The prefix for each concatenated element will be f'{prefix}_combined_'.

        Args:
        - prompt_dict: dict of the prompt text, tokens, attention mask, etc.
        - generation_dict: dict of the generation text, tokens, attention mask, etc.
        - prefix: str to prepend to the keys of the tokenized (prompt + generation)

        Returns:
            A dict of the (prompt + generation) text, tokens, attention mask, etc, along with the labels for the
            joint sequence, where the prompt token labels have been set to -100.
        """
        combined_dict = { f'{prefix}_combined_text' : prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text'] }

        for k,v in self.tokenizer(prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text']).items():
            combined_dict[f'{prefix}_combined_{k}'] = v

        combined_dict[f'{prefix}_labels'] = combined_dict[f'{prefix}_combined_input_ids'][:]  # contains both input and response (unpadded)
        combined_dict[f'{prefix}_labels'][:len(prompt_dict['prompt_input_ids'])] = [-100] * len(prompt_dict['prompt_input_ids'])

        return combined_dict

    def __iter__(self):
        """Create a flat version of the data and yield batches."""
        raise NotImplementedError
    

class SFTDataLoader(DataLoader):
    """
    Dataloader for supervised fine-tuning.
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        if self.split == "train":
            random.shuffle(prompts)  # otherwise, will be frontloaded with prompts in same domain

        for prompt in prompts:
            # flat_data.append(self.full_data[prompt])  # todo: change
            for gen in self.full_data[prompt].generations:
                tmp = copy.deepcopy(self.full_data[prompt])
                tmp.generations = [gen]
                flat_data.append(tmp)
        print(f"===============flat data with prompt repeat (split: {self.split}): {len(flat_data)}==========================")

        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            if "train" in self.split:
                random.shuffle(flat_data)

            batch = []

            for ei, example in enumerate(flat_data):
                batch_element = self.tokenize_batch_element(
                    # control token will be None for all losses other than csft
                    example.prompt + (self.kwargs.get('chosen_control_token') or ''),
                    example.generations[example.sft_index],
                    example.truncation_mode
                )
                batch_element['original_prompt'] = example.original_prompt
                batch.append(batch_element)

                if len(batch) == self.batch_size:  # todo:  ei == len(flat_data)-1
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished generating {self.n_examples} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                # if len(batch)>=1:
                #     yield self.collate(batch)
                done = True


class UnpairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do not require pairwise preferences (e.g., KTO).

    Since all the datasets have (or imply) pairwise preferences, this function assumes all preferred/dispreferred
    generations are from the desirable/undesirable conditional generations given x. 
    """
    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        """
        if self.max_prompt_count:
            num_unique = sum(min(self.max_prompt_count, len(self.full_data[prompt].pairs)) for prompt in prompts)
        else:
            num_unique = sum(len(self.full_data[prompt].pairs) for prompt in prompts)

        allowed_desirable = num_unique * self.kwargs.get('frac_unique_desirable', 1.0)
        allowed_undesirable = num_unique * self.kwargs.get('frac_unique_undesirable', 1.0)
        seen_desirable = 0  # how much positive samples
        seen_undesirable = 0 # how much negative samples

        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for i,j in example.pairs:
                if seen_desirable < allowed_desirable:
                    flat_data.append((example, example.generations[i], 'chosen'))
                    seen_desirable += 1

                '''
                flat_data[0]
                (Example(
                prompt='\n<|user|>\nWhat do I do if my child is always yelling at me?\n<|assistant|>\n', 
                generations=['You could consider what behaviors you find frustrating, and what the underlying emotions might be that cause them. Also, consider how you might respond differently next time.', 
                'Do you want to know what to do about this?'], sft_index=0, scores=[], pairs=[(0, 1)], desirable=[], truncation_mode='keep_end', dataset_name='hh', original_prompt=''), 
                'You could consider what behaviors you find frustrating, and what the underlying emotions might be that cause them. Also, consider how you might respond differently next time.', 
                'chosen')
                '''
                
                if seen_undesirable < allowed_undesirable:
                    flat_data.append((example, example.generations[j], 'rejected'))
                    seen_undesirable += 1

                '''
                (Example(prompt='\n<|user|>\nWhat do I do if my child is always yelling at me?\n<|assistant|>\n', generations=['You could consider what behaviors you find frustrating, and what the underlying emotions might be that cause them. Also, consider how you might respond differently next time.', 'Do you want to know what to do about this?'], sft_index=0, scores=[], pairs=[(0, 1)], desirable=[], truncation_mode='keep_end', dataset_name='hh', original_prompt=''), 
                'Do you want to know what to do about this?', 
                'rejected')
                '''

        return flat_data

    def __iter__(self):
        prompts = list(self.full_data.keys())
        if self.split == "train":
            random.shuffle(prompts) # otherwise, will be frontloaded with prompts in same domain
        flat_data = self.get_flat_data(prompts)

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)   # so generations in the same preference are not in the same batch
            batch = []
            example_queue = []

            for example, generation, status in flat_data:
                batch_element = self.tokenize_batch_element(example.prompt, generation, example.truncation_mode, prefix='target')
                batch_element['status'] = status 
                batch_element['truncation_mode'] = example.truncation_mode
                '''
                dict_keys(['prompt_text', 'target_text', 'prompt_input_ids', 'prompt_attention_mask', 'target_input_ids', 'target_attention_mask', 
                'target_combined_text', 'target_combined_input_ids', 'target_combined_attention_mask', 
                'target_labels', 'status', 'truncation_mode'])
                '''

                example_queue.append(batch_element)
                
                if len(example_queue) >= self.batch_size:
                    while len(batch) < self.batch_size:
                        batch.append(example_queue.pop(0))  # pop() 把 example_queue 清空
                    
                if len(batch) >= self.batch_size:
                    # for estimating the KL term, match up x and y' that are not corresponding input-output pairs in the data
                    # for x_i, get a mismatched y' by just picking the subsequent y_{i+1} in the batch (desirable/undesirable status does not matter)
                    # the respective input IDs, attention mask, and so on will be prefixed by the term KL
                    indices = list(range(1, len(batch))) + [0]
                    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
                    for i in range(len(batch)):
                        batch[i].update(self.tokenize_batch_element(
                            batch[i]['prompt_text'],
                            batch[indices[i]]['target_text'],
                            batch[i]['truncation_mode'],
                            prefix='KL'
                        ))

                    example_idx += len(batch)
                    '''
                    batch[0].keys()
                    dict_keys(['prompt_text', 'target_text', 'prompt_input_ids', 'prompt_attention_mask', 'target_input_ids', 'target_attention_mask', 
                    'target_combined_text', 'target_combined_input_ids', 'target_combined_attention_mask', 
                    'target_labels', 'status', 'truncation_mode', 
                    'KL_text', 'KL_input_ids', 'KL_attention_mask', 
                    'KL_combined_text', 'KL_combined_input_ids', 'KL_combined_attention_mask', 'KL_labels'])
                    '''
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break


class UnpairedQueryPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do not require pairwise preferences (e.g., KTO).

    Since all the datasets have (or imply) pairwise preferences, this function assumes all preferred/dispreferred
    generations are from the desirable/undesirable conditional generations given x.
    """

    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        """
        if self.max_prompt_count:
            num_unique = sum(min(self.max_prompt_count, len(self.full_data[prompt].pairs)) for prompt in prompts)
        else:
            num_unique = sum(len(self.full_data[prompt].generations) for prompt in prompts)

        print("num_unique: ", num_unique)
        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            for i in range(len(example.generations)):
                flat_data.append((example, example.generations[i], example.desirable[i]))


        return flat_data

    def __iter__(self):
        prompts = list(self.full_data.keys())
        if self.split == "train":
            random.shuffle(prompts)  # otherwise, will be frontloaded with prompts in same domain
        flat_data = self.get_flat_data(prompts)

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)  # so generations in the same preference are not in the same batch
            batch = []
            example_queue = []

            for example, generation, status in flat_data:
                batch_element = self.tokenize_batch_element(example.prompt, generation, example.truncation_mode,
                                                            prefix='target')
                batch_element['status'] = status
                batch_element['truncation_mode'] = example.truncation_mode
                '''
                dict_keys(['prompt_text', 'target_text', 'prompt_input_ids', 'prompt_attention_mask', 'target_input_ids', 'target_attention_mask', 
                'target_combined_text', 'target_combined_input_ids', 'target_combined_attention_mask', 
                'target_labels', 'status', 'truncation_mode'])
                '''

                example_queue.append(batch_element)

                if len(example_queue) >= self.batch_size:
                    while len(batch) < self.batch_size:
                        batch.append(example_queue.pop(0))  # pop() 把 example_queue 清空

                if len(batch) >= self.batch_size:
                    # for estimating the KL term, match up x and y' that are not corresponding input-output pairs in the data
                    # for x_i, get a mismatched y' by just picking the subsequent y_{i+1} in the batch (desirable/undesirable status does not matter)
                    # the respective input IDs, attention mask, and so on will be prefixed by the term KL
                    indices = list(range(1, len(batch))) + [0]
                    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
                    for i in range(len(batch)):
                        batch[i].update(self.tokenize_batch_element(
                            batch[i]['prompt_text'],
                            batch[indices[i]]['target_text'],
                            batch[i]['truncation_mode'],
                            prefix='KL'
                        ))

                    example_idx += len(batch)
                    '''
                    batch[0].keys()
                    dict_keys(['prompt_text', 'target_text', 'prompt_input_ids', 'prompt_attention_mask', 'target_input_ids', 'target_attention_mask', 
                    'target_combined_text', 'target_combined_input_ids', 'target_combined_attention_mask', 
                    'target_labels', 'status', 'truncation_mode', 
                    'KL_text', 'KL_input_ids', 'KL_attention_mask', 
                    'KL_combined_text', 'KL_combined_input_ids', 'KL_combined_attention_mask', 'KL_labels'])
                    '''
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break


class PairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do require pairwise preferences (e.g., DPO).
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        if "train" in self.split:
            random.shuffle(prompts)  # otherwise, will be frontloaded with prompts in same domain

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for pair in example.pairs:
                flat_data.append((example, pair))
         
        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            if self.split == "train":
                random.shuffle(flat_data)
            batch = []

            for ei, (example, (i,j)) in enumerate(flat_data):
                batch_element = {}
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[i], example.truncation_mode, prefix='chosen'))
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[j], example.truncation_mode, prefix='rejected'))
                batch.append(batch_element)

                if len(batch) >= self.batch_size:  # todo:  or ei == len(flat_data)-1
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished {example_idx} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                # if len(batch) >= 1:
                #     yield self.collate(batch)  # todo, mathdpo error
                done = True
