#!/bin/sh
model_path=/cache/ckpt/Qwen2.5-Math-7B-Instruct

input_file=$1
output_file=$2
mode=$3

start_idx=0
end_idx=0
step=5000
master_port=29405
#gpu_list=(0)
gpu_list=(0 1 2 3 4 5 6 7)


for i in $(seq 0 `expr ${#gpu_list[@]} - 1`); do
  gpu=${gpu_list[i]}
  echo "using gpu ${gpu}; file: ${input_file}"
  log_file="l${gpu}.log"
  start_idx=$end_idx
  end_idx=`expr $start_idx + $step`
  echo "start idx: "${start_idx}" end idx: "${end_idx}
  master_port=`expr $master_port + 1`

  nohup python relabel.py \
    ++use_gpus=${gpu} \
    ++mode=${mode} \
    ++saved_policy=${model_path} \
    ++test_file=${input_file} \
    ++output_file=${output_file} \
    ++torch_type="float16" \
    ++temperature=0 \
    ++top_p=1.0 \
    ++model.max_new_length=2048 \
    ++start_idx=${start_idx} \
    ++end_idx=${end_idx} > ${log_file} 2>&1 &
done
