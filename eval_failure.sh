#!/bin/sh

metric=$1
input_file=$2
output_file=$3

LLAMA_GUARD_CKPT=/cache/ckpt/llama-guard-2-8b
LLAMA_GUARD_CKPT_HF=/cache/ckpt/Meta-Llama-Guard-2-8B
QWEN_CKPT=/cache/ckpt/Qwen2.5-Math-7B-Instruct


start_idx=0
end_idx=0
master_port=29405
gpu=0
log_file=l0.log

if [ "${metric}" == "safety" ]; then

  torchrun --nproc_per_node 1 --master_port ${master_port} eval.py \
  ++use_gpus=${gpu} \
  ++mode=${metric} \
  ++saved_policy=${LLAMA_GUARD_CKPT} \
  ++model.tokenizer_name_or_path=${LLAMA_GUARD_CKPT_HF} \
  ++input_file=${input_file} \
  ++output_file=${output_file} \
  ++batch_size=1 \
  ++temperature=0 \
  ++prompt_key="predict_instruction" \
  ++target_key="target_model_response" \
  ++top_p=1.0 \
  ++model.max_new_length=2048 > ${log_file} 2>&1 &

elif [ "${metric}" == "self_bleu" ]; then

  torchrun --nproc_per_node 1 --master_port ${master_port} eval.py \
  ++use_gpus=${gpu} \
  ++mode=${metric} \
  ++saved_policy=${LLAMA_GUARD_CKPT} \
  ++model.tokenizer_name_or_path=${LLAMA_GUARD_CKPT_HF} \
  ++input_file=${input_file} \
  ++output_file=${output_file} \
  ++batch_size=1 \
  ++temperature=0 \
  ++prompt_key="predict_instruction" \
  ++target_key="target_model_response" \
  ++top_p=1.0 \
  ++model.max_new_length=2048 > ${log_file} 2>&1 &

else
  nohup python eval.py \
    ++use_gpus=${gpu} \
    ++mode=${metric} \
    ++saved_policy=${QWEN_CKPT} \
    ++input_file=${input_file} \
    ++output_file=${output_file} \
    ++prompt_key="predict_instruction" \
    ++target_key="target_model_response" \
    ++torch_type="float16" \
    ++temperature=0 \
    ++top_p=1.0 \
    ++model.max_new_length=2048 > ${log_file} 2>&1 &

done
