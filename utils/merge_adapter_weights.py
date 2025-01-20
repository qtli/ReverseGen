import pdb

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', type=str, help='input file')
parser.add_argument('--merge_path', type=str, help='input file')
parser.add_argument('--prev_path', type=str, help='input file')
parser.add_argument('--check_para', action="store_true", help='input file')

args = parser.parse_args()

if os.path.exists(args.merge_path) is False:
    os.mkdir(args.merge_path)
    # lora_merge_model = AutoPeftModelForCausalLM.from_pretrained(
    #     args.lora_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float32,
    # )
    # # Merge LoRA and base model and save
    # lora_merge_model = lora_merge_model.merge_and_unload()
    base_model = AutoModelForCausalLM.from_pretrained(args.prev_path,
                                                      low_cpu_mem_usage=True,
                                                      torch_dtype=torch.float32)
    lora_model = PeftModel.from_pretrained(base_model,
                                           args.lora_path,
                                           torch_dtype=torch.float32)
    lora_merge_model = lora_model.merge_and_unload()

    lora_merge_model.save_pretrained(args.merge_path, safe_serialization=True, max_shard_size="8GB")

    target_model_path = os.path.split(args.merge_path)[0]

    # Move all files from the source directory to the target model path
    for filename in os.listdir(args.merge_path):
        source_file = os.path.join(args.merge_path, filename)
        destination_file = os.path.join(target_model_path, filename)

        # Move the file
        shutil.move(source_file, destination_file)

else:
    print("merge file exist!")
    if args.check_para:
        print("check parameters")
        lora_merge_model = AutoModelForCausalLM.from_pretrained(args.merge_path, low_cpu_mem_usage=True, use_safetensors=True)


if args.check_para:
    try:
        prev_model = torch.load(args.prev_path, map_location='cpu')["state"]
        total_params_prev_model = sum(prev_model[n].numel() for n in prev_model.keys())
    except Exception as e:
        prev_model = AutoModelForCausalLM.from_pretrained(args.prev_path, low_cpu_mem_usage=True, use_safetensors=True)
        total_params_prev_model = sum(p.numel() for p in prev_model.parameters())
        prev_model = prev_model.state_dict()

    total_params_lora_merge_model = sum(p.numel() for p in lora_merge_model.parameters())

    print("total_params_lora_merge_model: ", total_params_lora_merge_model)
    print("total_params_prev_model: ", total_params_prev_model)

    for name, param in lora_merge_model.named_parameters():
        if "gate_proj" in name or "q_proj" in name or "o_proj" in name:
            print("merged parameter:")
            # if name in prev_model:
            if name in prev_model:
                print(name)
                # print(param - prev_model[name])
                print((param - prev_model[name]).sum())
                # pdb.set_trace()
            else:
                print("===========not exist: ", name)
                pdb.set_trace()


'''
python merge_adapter_weights.py \
--merge_path /cache/ckpt/reversegen_rebuttal/llama-8b-instuct-as-proposer-warmup-difficult-data-gsm \
--prev_path /cache/ckpt/llama-3-8B-instruct/ \
--check_para

python merge_adapter_weights.py \
--lora_path /cache/ckpt/reversegen_rebuttal/llama-8b-as-proposer-warmup-difficult-data-gsm/LORA \
--merge_path /cache/ckpt/reversegen_rebuttal/llama-8b-as-proposer-warmup-difficult-data-gsm/MERGE \
--prev_path /cache/ckpt/Meta-Llama-3-8B

python merge_adapter_weights.py \
--merge_path /cache/ckpt/reversegen_rebuttal/llama-8b-as-proposer-warmup-difficult-data-gsm \
--prev_path /cache/ckpt/Meta-Llama-3-8B \
--check_para
'''