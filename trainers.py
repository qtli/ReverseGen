# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Extendable Trainer classes for aligning LLMs.
The specific class that should be used should be specified in the loss file under config/loss.

The BasicTrainer contains the core methods (e.g., sharding, basic training loop, etc.).
The SFTTrainer, PairedPreferenceTrainer, and UnpairedPreferenceTrainer all subclass BasicTrainer
and override the get_batch_metrics() and (optionally) forward() methods.

The trainer for each loss should subclass either PairedPreferenceTrainer or UnpairedPreferenceTrainer.
"""
import pdb

import torch
torch.backends.cuda.matmul.allow_tf32 = True

import gc
from utils.models import AutoModelForCausalLMWithValueHead
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
import contextlib

import dataloader
from utils.common_funcs import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_batch_logps,
    masked_mean,
    masked_var,
    entropy_from_logits,
    delete_dict,
    rowwise_product,
    detect_abnormal_value
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


class BasicTrainer(object):
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 config: DictConfig, 
                 train_iterator: dataloader.DataLoader, 
                 eval_iterator: dataloader.DataLoader, 
                 policy: nn.Module, 
                 reference_model: Optional[nn.Module] = None, 
                 rank: int = 0, 
                 world_size: int = 1, 
                 fsdp: bool = False,
                 ):
        """A trainer for a language model, supporting either SFT, HALO, or offline PPO training.
        """
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.rank = rank
        self.device = torch.device('cuda', self.rank)
        self.world_size = world_size
        self.config = config
        self.run_dir = config.local_run_dir
        self.fsdp = fsdp

        self.tokenizer = tokenizer
        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)
        self.reference_model = reference_model
        self.example_counter = 0
        self.batch_counter = 0

        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator

        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.model.eval_batch_size}')

        if self.fsdp:
            self.shard()

        self.is_mistral = 'mistral' in self.config.model.name_or_path.lower()

    def shard(self):
        """
        Shard the policy model and reference model (if applicable) using FDSP.
        """
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        assert self.config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'
        wrap_class = get_block_class_from_model(self.policy.pretrained_model if self.config.loss.name == 'ppo' else self.policy, self.config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=True if self.config.enable_lora else False,
            sync_module_states=False
        )

        rank0_print('Sharding models...')
        mp_dtype = getattr(torch, self.config.model.fsdp_policy_mp) if self.config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)

        if self.config.loss.name == 'ppo':
            self.policy.pretrained_model = FSDP(self.policy.pretrained_model, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

            # shard the value head according to size
            v_head_shared_fsdp_kwargs = dict(
                auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=100),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=False),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.rank,
                ignored_modules=None,
                limit_all_gathers=False,
                use_orig_params=False,
                sync_module_states=False
            )
            self.policy.v_head = FSDP(self.policy.v_head, **v_head_shared_fsdp_kwargs)
        else:
            self.policy = FSDP(self.policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if self.reference_model is not None:
            self.reference_model = FSDP(self.reference_model, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if self.config.model.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')

                if self.config.loss.name == 'ppo':
                    apply_activation_checkpointing(self.policy.pretrained_model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
                else:
                    apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

                if self.reference_model is not None:
                    apply_activation_checkpointing(self.reference_model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

                rank0_print('FSDP activation checkpointing enabled!')

        if self.config.enable_lora:
            print('Loaded model on rank', self.rank)
            self.policy.enable_input_require_grads()  # todo

        dist.barrier()
            
    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy."""
        # https://github.com/pytorch/pytorch/issues/100069
        random.seed(42)
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if self.fsdp else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                # max_length=self.config.model.max_length,  # prompt + max_new_token
                max_new_tokens=self.config.model.max_new_length,
                do_sample=True if int(self.config.top_p) != 1 else False,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
                num_return_sequences=self.config.num_try,
            )
            policy_output = pad_to_length(policy_output, self.config.model.max_length, self.tokenizer.pad_token_id)
            policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
            policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
            rank0_print("policy_output_decoded: ", policy_output_decoded)

        hier_policy_output_decoded = []
        bsz = batch['prompt_input_ids'].size(0)
        for i in range(bsz):
            hier_policy_output_decoded.append(policy_output_decoded[i*self.config.num_try: (i+1)*self.config.num_try])
        return hier_policy_output_decoded


    def get_batch_samples_logits(self, batch: Dict[str, torch.LongTensor], gen_uncertainty: bool) -> Tuple[str, str]:
        """Generate samples from the policy."""
        # https://github.com/pytorch/pytorch/issues/100069
        random.seed(42)

        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if self.fsdp else contextlib.nullcontext())
        SURE = []
        UNSURE = []
        SURE.append(self.tokenizer("sure")['input_ids'][1])  # 0 refers to the start idx
        UNSURE.append(self.tokenizer("unsure")['input_ids'][1])

        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_new_tokens=self.config.model.max_new_length,
                output_scores=True,
                num_return_sequences=self.config.num_try,
                return_dict_in_generate=True
            )
            if gen_uncertainty:
                # Use model trained on MMLU QA and certainty data, to answer the certainty
                logits = policy_output['scores']

                # greedy decoding and calculate the confidence of sure and unsure
                pt = torch.softmax(torch.Tensor(logits[0][0]), dim=0)
                sure_prob = pt[SURE[0]]
                unsure_prob = pt[UNSURE[0]]
                sure_prob = sure_prob / (sure_prob + unsure_prob)  # normalization
                sure_prob_float = [[sure_prob.cpu().numpy().tolist()]]
                # todo: batch_size > 1
                return sure_prob_float, ""
            else:
                logits = policy_output['scores'][0][0]  # The first token
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[self.tokenizer("A").input_ids[1]],  # index 0 - start token
                                logits[self.tokenizer("B").input_ids[1]],
                                logits[self.tokenizer("C").input_ids[1]],
                                logits[self.tokenizer("D").input_ids[1]],
                            ]
                        ),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                output_text = [{0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]]
                conf = [np.max(probs).item()]

                return output_text, conf


    def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the losses, one for each example (sif chosen_only or rejected_only, only n/2 losses).
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively, for reporting.
            Note that rejected responses do not factor into the loss, only the reward calculation.
        """
        raise NotImplementedError

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.
        
        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'
        """
        raise NotImplementedError


    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single explore_dataset.
        It does not compare two models to eacch other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        rank0_print(f'Running evaluation')
        self.policy.eval()

        if self.reference_model is not None:
            self.reference_model.eval()

        all_eval_metrics = defaultdict(list)
        
        for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
            print("self.world_size: ", self.world_size)
            pdb.set_trace()
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            pdb.set_trace()
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, mode='eval')

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        results = {
            'metadata': OmegaConf.to_object(self.config),
            'results': formatted_dict(mean_eval_metrics),
        }
        return results


    def sample(self, gen_mmlu_train=False, gen_uncertainty=False, fn=None) -> List[Dict[str, str]]:
        """
        Generate samples from the policy model.
        
        gen_mmlu_train: True if use model trained on MMLU QA and certainty data. Output: answer the option (A/B/C/D) and confidence
        gen_uncertainty: True if use model trained on MMLU QA and uncertainty data. Output: confidence based on sure/unsure
        """
        rank0_print(f"Writing samples to {fn}...")
        if os.path.exists(fn):
            sample_idx = json.loads(open(fn, 'r', encoding='utf-8').readlines()[-1])["idx"]
            sample_idx += 1
        else:
            sample_idx = 0
        wf = open(fn, 'a+', encoding='utf-8')

        all_policy_samples, all_prompts, all_chosen, all_original_prompts = [], [], [], []
        all_confs = []

        self.policy.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        for eval_batch in (tqdm.tqdm(self.eval_batches, desc='iterate eval batch') if self.rank == 0 else self.eval_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)

            if gen_mmlu_train:
                # Use model trained on MMLU QA and certainty data, to answer the certainty
                policy_samples, conf = self.get_batch_samples_logits(local_eval_batch, gen_uncertainty=gen_uncertainty)
                all_confs.extend(conf)
            else:
                policy_samples = self.get_batch_samples(local_eval_batch)
            chosen_samples = []

            # for DPO-like losses, chosen_text is the field that will contain the text; target_text for all other losses
            # be sure to remove EOS token if present
            for x in (eval_batch['target_text'] if 'target_text' in eval_batch else eval_batch['chosen_text']):
                if self.tokenizer.eos_token in x:
                    x = x[:x.rfind(self.tokenizer.eos_token)]
                chosen_samples.append(x)

            all_prompts.extend(eval_batch['prompt_text'])
            all_original_prompts.extend(eval_batch['original_prompt'])
            all_chosen.extend(chosen_samples)
            all_policy_samples.extend(policy_samples)

            if self.config.n_samples is not None and len(all_prompts) > self.config.n_samples:
                break
            else:
                rank0_print(f"Generated {len(all_prompts)} samples ...")

            for si, p in enumerate(eval_batch['prompt_text']):
                if gen_mmlu_train:
                    model_prediction = [ps for ps in policy_samples[si]]
                else:
                    model_prediction = [ps[len(eval_batch['prompt_text'][si]):] for ps in policy_samples[si]]

                for mp in model_prediction:
                    each_sample = {
                        'idx': sample_idx,
                        'prompt': eval_batch['prompt_text'][si],
                        'chosen': chosen_samples[si],
                        self.config.target_key: mp,
                    }

                    if gen_mmlu_train and gen_uncertainty is False:
                        # probability of that option chosen by policy (A/B/C/D)
                        each_sample["conf"] = conf[si]

                    line = json.dumps(each_sample, ensure_ascii=False)
                    wf.write(line + '\n')

                sample_idx += 1


    def train(self):
        """Begin either SFT or HALO training, with periodic evaluation. This is subclassed when implementing PPO."""

        rank0_print(f'Using {self.config.optimizer} optimizer with learning rate {self.config.lr}')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        if self.reference_model is not None:
            self.reference_model.eval()

        last_log = None
        gradients_accumulated = 0
        batch_metrics = defaultdict(list)


        # print(f'Loaded {len(train_batches)} train batches of size {config.model.batch_size}')
        # rank0_print(f'Loaded {len(list(self.train_iterator))} train batches ..................')
        for batch_idx, batch in enumerate(self.train_iterator):
            if batch_idx % 50 == 0:
                rank0_print(f"finished {batch_idx} train batches")

            if batch_idx == 0:
                rank0_print("prompt_text: ", batch["prompt_text"][0])
                if "target_text" in batch:
                    rank0_print("target_text: ", batch["target_text"][0])
                    rank0_print("target_combined_text: ", batch["target_combined_text"][0])
                else:
                    rank0_print("chosen_combined_text", batch['chosen_combined_text'][0])
                    rank0_print("rejected_combined_text", batch['rejected_combined_text'][0])

            # EVALUATION
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, mode='eval')

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                    delete_dict(local_eval_batch)

                mean_eval_metrics = {}
                for k, v in all_eval_metrics.items():
                    if len(v) > 0:
                        mean_eval_metrics[k] = sum(v) / len(v)
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
               
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    elif self.config.intermediate_checkpoints:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)

                delete_dict(all_eval_metrics)
                delete_dict(mean_eval_metrics)

            #### TRAINING
            self.policy.train()

            start_time = time.time()
            
            local_microbatch = slice_and_move_batch_for_device(batch, self.rank, self.world_size, self.rank)
            loss, metrics = self.get_batch_metrics(local_microbatch)
            (loss / self.config.model.gradient_accumulation_steps).backward()
            for k, v in metrics.items():
                batch_metrics[k].extend(v)

            gradients_accumulated += 1
            
            if gradients_accumulated == self.config.model.gradient_accumulation_steps:
                grad_norm = self.clip_gradient()
                batch_metrics['grad_norm'].append(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                gradients_accumulated = 0

            step_time = time.time() - start_time
            examples_per_second = self.config.model.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            
            self.batch_counter += 1
            self.example_counter += self.config.model.batch_size

            delete_dict(local_microbatch)
            delete_dict(metrics)

            if gradients_accumulated == 0 and (last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs):
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                delete_dict(batch_metrics)
                delete_dict(mean_train_metrics)
                delete_dict(batch)
                batch_metrics = defaultdict(list)

                # explicitly empty cache if less than 100MB available
                r = torch.cuda.memory_reserved(self.rank)
                a = torch.cuda.memory_allocated(self.rank)

                if (r - a) / 1024 < 100:
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

        # if self.config.enable_lora:
        #     self.policy = self.policy.merge_and_unload()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters."""
        if self.fsdp:
            return self.policy.clip_grad_norm_(self.config.model.max_grad_norm).item()
        # 梯度裁剪 对所有梯度 * clip_coef (<1) 目的是解决梯度爆炸问题，不解决梯度消失问题 clip_coef = max_norm / total_norm
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.model.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk.
        
        Args:
            step : current training step
            state: current state of training (model or optimizer, if applicable)
            metrics: dictionary of metrics to save
            dir_name: directory in which to save
        """
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None, save_model_only: bool=True):
        """
        Save tokenizer, policy model, optimizer, scheduler state to disk, gathering from all processes 
        and saving only on the rank 0 process.
        """
        # We don't need parallel processing. We only need one process and cache, other processes share.
        if self.fsdp:
            # allow save the state_dict on rank 0 and offload to CPU
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
                policy_state_dict = self.policy.state_dict()

                if self.config.enable_lora:
                    dir_name = os.path.join(self.run_dir, f'LORA')
                    os.makedirs(dir_name, exist_ok=True)
                    self.policy.save_pretrained(dir_name)

                    # todo: additional codes,
                    config_path = os.path.join(self.run_dir, 'config.yaml')
                    with open(config_path, 'w') as f:
                        OmegaConf.save(self.config, f)
                    rank0_print(f'saved lora model to {dir_name} ....')

            if self.rank == 0:
                if not self.config.enable_lora:
                    self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
                self.tokenizer.save_pretrained(self.run_dir)  # save tokenizer in HF format

            del policy_state_dict
            # rank != 0/-1 wait until all processes be here and main process have finished data processing
            dist.barrier()

            if not save_model_only:
                save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
                    optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

                if self.rank == 0:
                    self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
                del optimizer_state_dict
                dist.barrier()

                if self.rank == 0:
                    scheduler_state_dict = self.scheduler.state_dict()
                    self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
                del scheduler_state_dict
                dist.barrier()
        else:
            self.tokenizer.save_pretrained(self.run_dir) # save tokenizer in HF format
            policy_state_dict = self.policy.state_dict()
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
            del policy_state_dict

            if not save_model_only:
                optimizer_state_dict = self.optimizer.state_dict()
                self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
                del optimizer_state_dict

                scheduler_state_dict = self.scheduler.state_dict()
                self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
                del scheduler_state_dict
        

class SFTTrainer(BasicTrainer):
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}
        if mode is None: mode = self.config.mode

        policy_chosen_logits = self.policy(batch['target_combined_input_ids'],
                                           attention_mask=batch['target_combined_attention_mask'],
                                           use_cache=(not self.is_mistral)).logits.to(self.policy_dtype)
        policy_chosen_logps = get_batch_logps(policy_chosen_logits,
                                              batch['target_labels'],
                                              average_log_prob=False)
        losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)

        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps.float().cpu().numpy().tolist()
        metrics[f'loss/{mode}'] = all_devices_losses.float().cpu().numpy().tolist()
        rank0_print(metrics)

        return losses.mean(), metrics


class UnpairedPreferenceTrainer(BasicTrainer):
    """A trainer for any loss that doesn't use paired preference, like KTO."""
    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], average_log_prob=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        """Run the given model on the given batch of inputs.
        
        Returns:
            chosen_logps: log probabilities of chosen examples (should be batch size / 2 if data was read in correctly)
            rejected_logps: log probabilities of rejected examples (should be batch size / 2 if data was read in correctly)
        """
        all_logits = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask'], use_cache=(not self.is_mistral)).logits.to(self.policy_dtype)
        all_logps = get_batch_logps(all_logits, batch['target_labels'], average_log_prob=average_log_prob)

        assert all_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(all_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(all_logps.shape[0]) if batch['status'][i] == 'rejected']

        chosen_logps = all_logps[chosen_idx, ...]
        rejected_logps = all_logps[rejected_idx, ...]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        if mode is None: mode = self.config.mode

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

        # all_gather treats empty lists/tensors poorly, and empty lists can occur because a batch can contain all chosen or all rejected example
        # therefore, concatenate chosen + rejected rewards before all_gather
        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards)).to(self.device)

        all_rewards = all_gather_if_needed(combined_rewards, self.rank, self.world_size)
        all_statuses = all_gather_if_needed(combined_statuses, self.rank, self.world_size)
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx].float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx].float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/margins'] = [(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()]
        metrics[f'loss/{mode}'] = all_devices_losses.float().cpu().numpy().tolist()

        del policy_chosen_logps, policy_rejected_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_devices_losses

        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.mean(), metrics


class PairedPreferenceTrainer(BasicTrainer):
    """A trainer for any loss that uses paired preference, like DPO."""
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            
        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(batch['chosen_combined_input_ids'].shape[1], batch['rejected_combined_input_ids'].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        return concatenated_batch

    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], average_log_prob=False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
           Return two tensors of shape (batch size), one of the chosen examples, another of the rejected ones.

           Returns:
            chosen_logps: log probabilities of chosen examples (should be batch size / 2 if data was read in correctly)
            rejected_logps: log probabilities of rejected examples (should be batch size / 2 if data was read in correctly)
        """
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_combined_input_ids'], attention_mask=concatenated_batch['concatenated_combined_attention_mask'], use_cache=(not self.is_mistral)).logits.to(self.policy_dtype)
        all_logps = get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=average_log_prob)
        chosen_logps = all_logps[:batch['chosen_combined_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_combined_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        if mode is None: mode = self.config.mode

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch)
            print("policy_chosen_logps: ", torch.isnan(policy_chosen_logps).sum())
            print("policy_rejected_logps: ", torch.isnan(policy_rejected_logps).sum())
            print("reference_chosen_logps: ", torch.isnan(reference_chosen_logps).sum())
            print("reference_rejected_logps: ", torch.isnan(reference_rejected_logps).sum())
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

        # accuracy calculated on unpaired examples (for apples-to-apples comparison with UnpairedPreferenceTrainer)
        reward_accuracies = (chosen_rewards > rejected_rewards.flip(dims=[0])).float()

        chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)

        metrics[f'rewards_{mode}/chosen'] = chosen_rewards.float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/rejected'] = rejected_rewards.float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/accuracies'] = reward_accuracies.float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/margins'] = (chosen_rewards - rejected_rewards).float().cpu().numpy().tolist()
        metrics[f'logps_{mode}/rejected'] = policy_rejected_logps.float().cpu().numpy().tolist()
        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps.float().cpu().numpy().tolist()
        metrics[f'loss/{mode}'] = all_devices_losses.float().cpu().numpy().tolist()

        del chosen_rewards, rejected_rewards, reward_accuracies, policy_chosen_logps, policy_rejected_logps, all_devices_losses
        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.mean(), metrics


class DPOTrainer(PairedPreferenceTrainer):
       def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.config.loss.beta * logits)
        chosen_rewards = self.config.loss.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.config.loss.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


class KTOTrainer(UnpairedPreferenceTrainer):
    def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             policy_KL_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             reference_KL_logps) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        The desirable losses are weighed by config.loss.desirable_weight.
        The undesirable losses are weighed by config.loss.undesirable_weight.
        This should be used to address imbalances in the ratio of desirable:undesirable examples respectively.

        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x). Doing so avoids the requirement that there be equal numbers of
        desirable and undesirable examples in the microbatch.
        """
        # detach() 返回一个新tensor，从当前计算图中分离下来，但是仍指向原变量存放位置。
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # nn.all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        # 由于 master node 和 其他node的代码不同，all_reduce函数使用ring通讯，高效完成了组内变量的共享和计算
        # 第一个参数为需要进行运算的变量
        # 第二个参数op包含了一些方法
        if self.world_size > 1:
            dist.nn.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        # clamp(min, max) 将input tensor的每个元素夹紧在[min, max]区间
        KL = (KL / self.world_size).clamp(min=0)
        if len(KL.shape) == 0:
            KL = KL.unsqueeze(0) # make a one-dimensional tensor

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
            chosen_losses = 1 - F.sigmoid(self.config.loss.beta * (chosen_logratios - KL))
            chosen_rewards = self.config.loss.beta * chosen_logratios.detach()
        else:
            # batch里不存在chosen样本
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = (policy_rejected_logps - reference_rejected_logps)
            rejected_losses = 1 - F.sigmoid(self.config.loss.beta * (KL - rejected_logratios))
            rejected_rewards = self.config.loss.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.device)

        # desirable_weight: 1.0; undesirable_weight: 1.0
        losses = torch.cat((self.config.loss.desirable_weight * chosen_losses, self.config.loss.undesirable_weight * rejected_losses), 0)

        return losses, chosen_rewards, rejected_rewards, KL

    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], average_log_prob=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs. The examples used to calculate the rewards and the KL term should be
        processed in a single forward pass, since the gradient is taken wrt both groups. Doing it in multiple forward passes will give
        you a RuntimeError: 'The tensor has a non-zero number of elements, but its data is not allocated yet.'

        Args:
            - model: the model to use for the forward pass
            - batch: the microbatch (should have the input ids, attention mask, and labels)
            - average_log_prob: average the log probability across tokens in the output

        Returns:
            chosen_logps: log probabilities of chosen examples (should be batch size / 2 if data was read in correctly)
            rejected_logps: log probabilities of rejected examples (should be batch size / 2 if data was read in correctly)
            KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference; should be batch size)
        """
        max_length = max(batch['target_combined_input_ids'].shape[1], batch['KL_combined_input_ids'].shape[1])
        concatenated_batch = {}

        '''
        dict_keys(['prompt_text', 'target_text', 'prompt_input_ids', 'prompt_attention_mask', 'target_input_ids', 'target_attention_mask', 
            'target_combined_text', 'target_combined_input_ids', 'target_combined_attention_mask', 'target_labels', 
            'status', 'truncation_mode', 
            'KL_text', 'KL_input_ids', 'KL_attention_mask', 
            'KL_combined_text', 'KL_combined_input_ids', 'KL_combined_attention_mask', 'KL_labels'])
            
            
        batch["target_combined_input_ids"]: torch.Size([batch_size, 636])
        batch["KL_combined_input_ids"]: torch.Size([batch_size, 634])
        '''

        for k in batch:
            if k.startswith('target') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('target', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        for k in batch:
            if k.startswith('KL') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('KL', 'concatenated')
                # 把 target_combined_input_ids 与　KL_combined_input_ids　在batch 维度拼接
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        '''
        concatenated_batch:
        dict_keys(['concatenated_input_ids', 'concatenated_attention_mask', 
        'concatenated_combined_input_ids', 'concatenated_combined_attention_mask', 'concatenated_labels'])
        
        concatenated_batch["concatenated_input_ids"]: torch.Size([2 * batch_size, 636])        
        '''

        all_logits = model(
            concatenated_batch[f'concatenated_combined_input_ids'],
            attention_mask=concatenated_batch[f'concatenated_combined_attention_mask']
        ).logits.to(self.policy_dtype)  # torch.Size([batch_size * 2, 636, 32000])

        if detect_abnormal_value(all_logits):
            rank0_print("There is at least an NAN in all_logits !!!!")
            rank0_print("==================================")

        # Compute the log probabilities of the given labels under the given logits
        all_logps = get_batch_logps(all_logits, concatenated_batch[f'concatenated_labels'], average_log_prob=average_log_prob)
        # all_logps: torch.Size([batch_size * 2]) - by sum(-1) instance-level, not token-level


        # Split: target_logps are training data with 0、1 labels,for computing reward r_KTO(x,y)
        target_logps = all_logps[:batch['target_combined_input_ids'].shape[0]]  # torch.Size([batch_size])
        # KL_logps for computing reference points
        KL_logps = all_logps[batch['target_combined_input_ids'].shape[0]:]  # torch.Size([batch_size])


        if detect_abnormal_value(target_logps):
            rank0_print("There is at least an NAN in target_logps !!!!")
        if detect_abnormal_value(KL_logps):
            rank0_print("There is at least an NAN in KL_logps !!!!")

        rank0_print("target_logps: ", target_logps)
        rank0_print("KL_logps: ", KL_logps)

        assert target_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'rejected']
        chosen_logps = target_logps[chosen_idx, ...]
        rejected_logps = target_logps[rejected_idx, ...]

        return chosen_logps, rejected_logps, KL_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        if mode is None: mode = self.config.mode

        policy_chosen_logps, policy_rejected_logps, policy_KL_logps = self.forward(self.policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, reference_KL_logps = self.forward(self.reference_model, batch)

        losses, chosen_rewards, rejected_rewards, KL = self.loss(
            policy_chosen_logps, # torch.Size([M])
            policy_rejected_logps, # torch.Size([N])  M + N = batch_size
            policy_KL_logps, # torch.Size([batch_size])
            reference_chosen_logps, # torch.Size([M])
            reference_rejected_logps, # torch.Size([N])  M + N = batch_size
            reference_KL_logps # torch.Size([batch_size])
        )

        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards)).to(self.device)

        all_rewards = all_gather_if_needed(combined_rewards, self.rank, self.world_size)
        all_statuses = all_gather_if_needed(combined_statuses, self.rank, self.world_size)
        all_KL = all_gather_if_needed(KL, self.rank, self.world_size)
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx].float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx].float().cpu().numpy().tolist()
        metrics[f'rewards_{mode}/margins'] = [(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()]
        metrics[f'rewards_{mode}/KL_estimate'] = all_KL.float().cpu().numpy().tolist()
        metrics[f'loss/{mode}'] = all_devices_losses.float().cpu().numpy().tolist()
        del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_devices_losses, all_KL
        return losses.mean(), metrics


class PPOTrainer(BasicTrainer):
    """One-step, offline variant of PPO."""
    def forward(self, model: AutoModelForCausalLMWithValueHead, batch: Dict[str, Union[List, torch.LongTensor]], is_policy: bool=True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.

        Args:
            model: model to run forward pass on
            batch: input batch (forward pass will be run on keys with prefix 'chosen')
            masks: binary-valued tensor shape (batch size, sequence length)
            is_policy: whether the model is the policy or reference

        Returns: 
            all_logps: batch log probabilities at the token level of shape (batch size, sequence length)
            all_logits: corresponding logits of shape (batch size, sequence length)
            all_values: values predicted for each token, of shape (batch size, sequence length)
        """
        if is_policy:
            # here the prefix 'chosen' is a misnomer, since it can refer to the dispreferred generations
            # the 'status' field contains the actual status of the generations
            all_logits, _, all_values = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask'])
            all_values = all_values[:, :-1].contiguous().to(self.rank)
        else:
            all_logits = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask'], use_cache=(not self.is_mistral)).logits.to(self.policy_dtype)
            all_values = None

        all_logps = get_batch_logps(all_logits.to(self.policy_dtype), batch['target_labels'], average_log_prob=False, token_level=True)
        # Returned tensors will have sequence length that is one less than the inputs (to account for label shifting).
        all_logits = all_logits[:, :-1].contiguous().to(self.rank)
        all_logps = all_logps.contiguous().to(self.rank)

        return all_logps, all_logits, all_values
    
    def compute_advantages(self, values: torch.FloatTensor, rewards: torch.FloatTensor, masks: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Estimate the advantages and rewards for every token taken.

        Args:
            values: the estimated values of the tokens. Should already be detached from graph.
            rewards: signal from the environment as to whether the generation is good or bad.
                In the basic implementation, this is only one nonzero reward, on the last unpadded token of each sequence.
                torch tensor of shape (batch size, sequence length)
            masks: torch tensor of shape (batch size, sequence length); 1 if token should be considered and 0 otherwise

        Returns:
            advantages: torch tensor of shape (batch size, sequence length)
            returns: Also called 'rewards-to-go'.
                Only tokens after the current token are used to calculate this: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
                torch tensor of shape (batch size, sequence length)
        """
        values = values * masks
        rewards = rewards * masks
        gae = 0 # generalized advantage estimation
        seq_len = rewards.shape[-1]
        advantages_reversed = []
        
        discounted_future_reward = torch.zeros_like(rewards[:,0])
        discounted_future_rewards_reversed = []

        for t in reversed(range(seq_len)):
            # see https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
            delta = rewards[:, t] + self.config.loss.gamma * (values[:, t + 1] if t < seq_len - 1 else 0.0) - values[:, t]
            gae = delta + self.config.loss.gamma * self.config.loss.lam * gae
            advantages_reversed.append(gae)
            
            discounted_future_rewards_reversed.append(discounted_future_reward)
            discounted_future_reward = rewards[:, t] + self.config.loss.gamma * discounted_future_reward

        advantages = (torch.stack(advantages_reversed[::-1]).transpose(0, 1) * masks)
        returns = (advantages + values).contiguous().to(self.rank)
        discounted_future_rewards = (torch.stack(discounted_future_rewards_reversed[::-1]).transpose(0, 1) * masks).contiguous().to(self.rank)

        # normalizing advantages leads to more stable learning
        mean_adv, var_adv = masked_mean(advantages, masks), masked_var(advantages, masks)
        normalized_advantages = (advantages - mean_adv) * torch.rsqrt(var_adv + 1e-8)
        normalized_advantages = (normalized_advantages * masks).detach().contiguous().to(self.rank)

        return normalized_advantages, returns, discounted_future_rewards

    def loss(self, batch: Dict, episode: Dict) -> Tuple[torch.FloatTensor, Dict]:
        """
        Given the batch statistics and the current episode's values, calculate the loss and return some loss statistics.

        Args:
            batch: dictionary containing batch data (shoud have keys 'values', 'returns', 'advantages', 'logprobs', 'masks')
            episode: dictionary containing the episode data (should have keys 'logits', 'values', 'logprobs')

        Returns:
            loss: combined policy and critic loss of shape (1,)
            loss_stats: dictionary of episode/batch statistics
        """
        value_losses = (episode['values'] - batch['discounted_future_rewards'].detach()) ** 2
        critic_loss = 0.5 * masked_mean(value_losses, batch['masks'])
        
        ratio = torch.exp(episode['logprobs'] - batch['logprobs'])
        policy_losses = -batch['advantages'] * ratio
        policy_losses_clipped = -batch['advantages'] * torch.clamp(ratio, self.config.loss.cliprange, 1 / self.config.loss.cliprange)
        policy_loss = masked_mean(torch.max(policy_losses, policy_losses_clipped), batch['masks'])

        KL_penalty = masked_mean(batch['logprobs'] - episode['logprobs'], batch['masks'])

        loss = policy_loss + self.config.loss.critic_coef * critic_loss + self.config.loss.KL_coef * KL_penalty

        loss_stats = {
            'loss/total' : loss.detach(),
            'loss/critic' : critic_loss.detach(),
            'loss/policy' : policy_loss.detach(),
            'clipfrac/policy' : masked_mean(torch.gt(policy_losses_clipped, policy_losses).float(), batch['masks']).detach(),
            'loss/entropy' :  entropy_from_logits(episode['logits'], batch['masks']).detach(),
            'loss/policykl' : masked_mean(batch['logprobs'] - episode['logprobs'], batch['masks']).detach(),
            'loss/seqratio' : rowwise_product(ratio, batch['masks']).mean().detach(),
        }

        return loss, loss_stats

    def train(self):
        """Train with PPO."""
        rank0_print(f'Using {self.config.optimizer} optimizer with learning rate {self.config.lr}')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        self.policy.train()
        self.reference_model.eval()
        
        last_log = None
        batch_metrics = defaultdict(list)
        gradients_accumulated = 0

        for batch_idx, batch in enumerate(self.train_iterator):
            if batch_idx == 0:
                rank0_print(batch.keys())
                rank0_print("prompt_text: ", batch["prompt_text"][0])
                if "target_text" in batch:
                    rank0_print("target_text: ", batch["target_text"][0])
                    rank0_print("target_combined_text: ", batch["target_combined_text"][0])
                else:
                    rank0_print("chosen_combined_text", batch['chosen_combined_text'][0])
                    rank0_print("rejected_combined_text", batch['rejected_combined_text'][0])

            batch_size = len(batch['prompt_text'])
            # todo:
            batch["scores"] = batch["status"]
            # batch['scores'] = torch.Tensor([(1 if batch['status'][i] == 'chosen' else -1) for i in range(batch_size)])
            local_microbatch = slice_and_move_batch_for_device(batch, self.rank, self.world_size, self.rank)

            with torch.no_grad():
                masks = (local_microbatch['target_labels'][:, 1:] != -100).clone().to(self.policy_dtype).contiguous().to(self.rank)
                logprobs, _, _ = self.forward(self.reference_model, local_microbatch, is_policy=False)
                _, _, values = self.forward(self.policy, local_microbatch)
                
                rewards = torch.zeros_like(masks) 
                for row in range(rewards.shape[0]):
                    rewards[row, masks[row].nonzero()[-1]] += local_microbatch['scores'][row]

                rewards = rewards.contiguous().to(self.rank) * masks
                advantages, returns, discounted_future_rewards = self.compute_advantages(values, rewards, masks)
                
            global_sbatch_dict = {
                "target_combined_input_ids" : batch['target_combined_input_ids'],
                "target_labels" : batch['target_labels'],
                "target_combined_attention_mask" : batch['target_combined_attention_mask'],
                "logprobs": all_gather_if_needed(logprobs, self.rank, self.world_size).to(self.policy_dtype),
                "values": all_gather_if_needed(values, self.rank, self.world_size).to(self.policy_dtype),
                "masks": all_gather_if_needed(masks, self.rank, self.world_size),
                "advantages": all_gather_if_needed(advantages, self.rank, self.world_size),
                "returns": all_gather_if_needed(returns, self.rank, self.world_size),
                "discounted_future_rewards": all_gather_if_needed(discounted_future_rewards, self.rank, self.world_size),
            }
            
            start_time = time.time()

            for ppo_epoch in range(self.config.loss.ppo_epochs):
                loss, local_batch_metrics = self.get_batch_metrics(global_sbatch_dict, batch_size, mode='train')

                for k,v in local_batch_metrics.items():
                    batch_metrics[k].extend(v)

                (loss / (self.config.model.gradient_accumulation_steps)).backward()
                gradients_accumulated += 1
                
            if gradients_accumulated == self.config.model.gradient_accumulation_steps:
                grad_norm = self.clip_gradient()
                batch_metrics['grad_norm'].append(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                gradients_accumulated = 0

            self.batch_counter += 1
            self.example_counter += batch_size

            step_time = time.time() - start_time
            examples_per_second = batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)

            delete_dict(global_sbatch_dict)
            delete_dict(local_microbatch)
            delete_dict(local_batch_metrics)
            delete_dict(batch)
            del _, masks, logprobs, values, rewards, advantages, returns, discounted_future_rewards

            if gradients_accumulated == 0 and (last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs):
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                delete_dict(batch_metrics)
                delete_dict(mean_train_metrics)
                batch_metrics = defaultdict(list)    
                
                # explicitly empty cache if less than 100MB available
                r = torch.cuda.memory_reserved(self.rank)
                a = torch.cuda.memory_allocated(self.rank)

                if (r - a) / 1024 < 100:
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

    def get_batch_metrics(self, global_sbatch_dict: Dict, batch_size: int, mode:str=None):
        """
        Given a batch that has been processed in the outer loop of PPO, return the batch statistics and the loss.
        """
        if mode is None: mode = self.config.mode
        indices = torch.randperm(batch_size).tolist()
        shuffled_global_microbatch = {}

        for k, v in global_sbatch_dict.items():
            if isinstance(v, torch.Tensor):
                v = v[indices]
            else:
                v = [ v[i] for i in indices ]
            
            shuffled_global_microbatch[k] = v

        local_microbatch = slice_and_move_batch_for_device(shuffled_global_microbatch, self.rank, self.world_size, self.rank)
        episode_logprobs, episode_logits, episode_values = self.forward(self.policy, local_microbatch)
        episode = {
            'logprobs' : episode_logprobs,
            'logits' : episode_logits,
            'values' : episode_values,
        }
        loss, metrics = self.loss(local_microbatch, episode)

        metrics['returns/mean'] = masked_mean(local_microbatch['returns'], local_microbatch['masks']).detach()
        metrics['returns/var'] = masked_var(local_microbatch['returns'], local_microbatch['masks']).detach()
        metrics['val/mean'] = masked_mean(local_microbatch['values'], local_microbatch['masks']).detach()
        metrics['val/var'] = masked_var(local_microbatch['values'], local_microbatch['masks']).detach()

        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())

        delete_dict(metrics)
        delete_dict(episode)
        delete_dict(local_microbatch)
        delete_dict(shuffled_global_microbatch)
        del episode_logprobs, episode_logits, episode_values

        return loss, batch_metrics

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        v_head_norm = torch.nn.utils.clip_grad_norm_(self.policy.v_head.parameters(), self.config.model.v_head_max_grad_norm).item()
        pretrained_model_norm =  torch.nn.utils.clip_grad_norm_(self.policy.pretrained_model.parameters(), self.config.model.max_grad_norm).item()
        return v_head_norm + pretrained_model_norm

    def save(self, output_dir=None, metrics=None, save_model_only=True):
        """
        Save tokenizer, policy, value head, optimizer, scheduler state to disk, gathering from all processes 
        and saving only on the rank 0 process.
        """
        # 这些操作不需要并行处理，只需要一个进程处理并缓存，其他进程共享缓存，
        if self.fsdp:
            # 设置一个阻塞栅栏，rank != 0/-1的进程等待，等到所有进程到达栅栏处，并且主进程处理数据结束，所有进程同步并释放。
            dist.barrier()
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.policy.pretrained_model, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
                policy_state_dict = self.policy.pretrained_model.state_dict()
                v_head_state_dict = self.policy.v_head.state_dict()

            if self.rank == 0:
                self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
                self.write_state_dict(self.example_counter, v_head_state_dict, metrics, 'v_head.pt', output_dir)
                self.tokenizer.save_pretrained(self.run_dir) # save tokenizer in HF format

            del policy_state_dict, v_head_state_dict
            dist.barrier()

            if not save_model_only:
                save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.policy.pretrained_model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
                    optimizer_state_dict = FSDP.optim_state_dict(self.policy.pretrained_model, self.optimizer)

                if self.rank == 0:
                    self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
                del optimizer_state_dict
                dist.barrier()

                if self.rank == 0:
                    scheduler_state_dict = self.scheduler.state_dict()
                    self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
                del scheduler_state_dict
                dist.barrier()
        else:
            self.tokenizer.save_pretrained(self.run_dir) # save tokenizer in HF format
            policy_state_dict = self.policy.pretrained_model.state_dict()
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
            del policy_state_dict

            if not save_model_only:
                optimizer_state_dict = self.optimizer.state_dict()
                self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
                del optimizer_state_dict

                scheduler_state_dict = self.scheduler.state_dict()
                self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
                del scheduler_state_dict
