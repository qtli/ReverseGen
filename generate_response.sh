#!/bin/sh

target_model_path=$1
test_file=$2
prompt_key=$3
response_output_file=$4


lora_train=false
if [ "${lora_train}" == "true" ]; then
  echo "MERGING ...."
  python utils/merge_adapter_weights.py \
  --lora_path ${target_model_path}/LORA \
  --merge_path ${target_model_path}/MERGE \
  --prev_path ${target_base_model_path}
fi


temperature=0
ns=0  # number of shots as in-context
nt=-1  # sampled responses for each instruction
gpu=0

nohup python infer.py \
  ++mode=respond \
  ++use_gpus=${gpu} \
  ++model.max_new_length=1024 \
  ++model.name_or_path=${target_model_path} \
  ++temperature=${temperature} \
  ++nshots=${ns} \
  ++num_try=${nt} \
  ++prompt_key=${prompt_key} \
  ++target_key="solution" \
  ++torch_type=float16 \
  ++test_file=${test_file} \
  ++output_file=${response_output_file} \
  ++saved_policy=${target_model_path} > ${log_file} 2>&1 &
