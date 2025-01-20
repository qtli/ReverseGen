# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main script for training.

Args:
- loss should have a file under config/loss that specifies the trainer in trainers.py and dataloader in dataloader.py
- model should have a file under config/model
- datasets is a list of datasets, each of which has a get_{name} function in dataloader.py
- exp_name is the experiment name (on WANDB); model will be saved to the cache_dir/exp_name
- model.load_from should be used for aligning a model that has already been finetuned

Remember to allocate enough RAM before running this (you need aroundd 800 GB for Llama-13B).
"""


import os
import hydra
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
import gc
from datetime import datetime
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import torch.multiprocessing as mp
from utils.models import AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import dataloader
from utils.common_funcs import disable_dropout, init_distributed, get_open_port, delete_dict, check_safetensor_exists
import pdb

def worker_main(rank: int, world_size: int, config: DictConfig, tokenizer: AutoTokenizer, train_iterator: dataloader.DataLoader, eval_iterator: dataloader.DataLoader, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer)."""
    if config.use_fsdp:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        wandb_dir = config.local_run_dir
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ['WANDB_CACHE_DIR'] = wandb_dir
        os.environ["WANDB_MODE"] = config.wandb.mode
        os.environ["WANDB_API_KEY"] = config.wandb.api_key

        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=wandb_dir,
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.loss.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(
        tokenizer, 
        config, 
        train_iterator, 
        eval_iterator, 
        policy, 
        reference_model=reference_model, 
        rank=rank, 
        world_size=world_size, 
        fsdp=config.use_fsdp,
    )
    start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(f'begin the training {start_time}')

    trainer.train()

    trainer.save()
    end_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(f'end the training {end_time}')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    if config.use_gpus != "all":
        print("use GPU: ", config.use_gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.use_gpus

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    os.makedirs(config.local_run_dir, exist_ok=True)
    print("Making experiment directory", config.local_run_dir)
    
    set_seed(config.seed)

    if config.eval_every % config.model.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.model.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.model.batch_size

    if config.use_fsdp and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    if config.saved_policy is None:
        config.saved_policy = f"{config.local_run_dir}/LATEST/policy.pt"

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)


    policy_kwargs = {'torch_dtype' : getattr(torch, config.model.policy_dtype)}
    reference_kwargs = {'torch_dtype' : getattr(torch, config.model.reference_dtype)}
    
    if not config.use_fsdp:
        policy_kwargs['device_map'] = 'balanced'
        reference_kwargs['device_map'] = 'balanced'
    use_safetensors = False
    for name in os.listdir(config.model.name_or_path):
        if name.endswith('.safetensors'):
            use_safetensors = True
            break
    model_class = AutoModelForCausalLMWithValueHead if config.loss.name == 'ppo' else AutoModelForCausalLM
    print("loading policy model from ", config.model.name_or_path)
    policy = model_class.from_pretrained(
        config.model.name_or_path, low_cpu_mem_usage=True, use_safetensors=use_safetensors, use_flash_attention_2=config.model.use_flash_attention, **policy_kwargs)
    disable_dropout(policy)

    if config.loss.use_reference_model:
        print('building reference model')
        reference_model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, low_cpu_mem_usage=True, use_safetensors=use_safetensors, use_flash_attention_2=config.model.use_flash_attention, **reference_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.load_from is not None:
        if check_safetensor_exists(config.model.load_from, ".safetensors"):
            state_dict = AutoModelForCausalLM.from_pretrained(config.model.load_from, low_cpu_mem_usage=True).state_dict()
            print(f'loading pre-trained weights from {config.model.load_from} ...')

        else:
            state_dict = torch.load(os.path.join(config.cache_dir, config.model.load_from), map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(f'loading pre-trained weights at step {step} from {config.model.load_from} with metrics {json.dumps(metrics, indent=2)}')

        if config.loss.name == 'ppo':
            policy.pretrained_model.load_state_dict(state_dict['state'])
            if config.loss.use_reference_model:
                reference_model.load_state_dict(state_dict['state'])
        else:
            if check_safetensor_exists(config.model.load_from, ".safetensors"):
                policy.load_state_dict(state_dict)
                if config.loss.use_reference_model:
                    reference_model.load_state_dict(state_dict)
            else:
                policy.load_state_dict(state_dict['state'])
                if config.loss.use_reference_model:
                    reference_model.load_state_dict(state_dict['state'])

        delete_dict(state_dict)
        gc.collect()
        torch.cuda.empty_cache()

        print('loaded pre-trained weights')

    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # add special tokens
    special_tokens = [config.loss.chosen_control_token, config.loss.rejected_control_token ] if config.loss.name == 'csft' else []
    num_added = tokenizer.add_special_tokens({ "additional_special_tokens" : special_tokens })
    print(f"{num_added} special tokens added")

    if special_tokens != []:
        if config.loss.name == 'ppo':
            # for PPO, policy and reference must tokenize the same way
            policy.pretrained_model.resize_token_embeddings(len(tokenizer))
            reference_model.resize_token_embeddings(len(tokenizer))
        else:
            policy.resize_token_embeddings(len(tokenizer))

    if config.enable_lora:
        print("===================Use LoRA========================")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters...")
        # policy.enable_input_require_grads() # https://github.com/huggingface/peft/issues/137
        policy = get_peft_model(policy, lora_config)

    data_loader_class = getattr(dataloader, config.loss.dataloader)
    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        human_prefix=config.human_prefix,
        human_suffix=config.human_suffix,
        assistant_prefix=config.assistant_prefix,
        assistant_suffix=config.assistant_suffix,
        seed=config.seed,
        start_idx=config.start_idx,
        end_idx=config.end_idx,
        frac_unique_desirable=config.frac_unique_desirable,
        frac_unique_undesirable=config.frac_unique_undesirable,
        # control tokens taken from Korbak et al.'s (2023) "Pretraining Models with Human Feedback"
        # SFTDataLoader will use them for sampling; ConditionalSFTDataLoader for training
        chosen_control_token=(config.loss.chosen_control_token if config.loss.name == "csft" else None),
        rejected_control_token=(config.loss.rejected_control_token if config.loss.name == "csft" else None),
    )

    train_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='train',
        batch_size=config.model.batch_size,
        n_epochs=config.n_epochs,
        # n_examples=config.n_examples,
        config=config,
        **data_iterator_kwargs
    )
    eval_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='eval',
        batch_size=config.model.eval_batch_size,
        n_examples=config.n_eval_examples,
        n_epochs=(1 if config.n_eval_examples is None else None),
        config=config,
        **data_iterator_kwargs
    )

    world_size = torch.cuda.device_count()
    if config.use_fsdp and world_size > 1:
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, tokenizer, train_iterator, eval_iterator, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, tokenizer, train_iterator, eval_iterator, policy, reference_model)


if __name__ == '__main__':
    main()
