import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, help='input file')
args = parser.parse_args()

if os.path.exists(os.path.join(args.ckpt_dir, "policy_vllm.pt")) is False:
    print("policy_vllm.pt does not exist!")
    state_dict = torch.load(os.path.join(args.ckpt_dir, "LATEST", "policy.pt"), map_location='cpu')["state"]
    torch.save(state_dict, os.path.join(args.ckpt_dir, "policy_vllm.pt"))
else:
    print("policy_vllm.pt exists!")

