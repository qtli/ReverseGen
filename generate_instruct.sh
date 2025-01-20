#!/bin/sh
conda activate reversegen

proposer_base_path=$1
exp_name=$2
use_vllm=$3
top_p=$4
test_file=$5
sample_file_mark=$6
start_idx=$7
end_idx=$8

ckpt_path=/cache/ckpt/reversegen/${exp_name}

if [ "${use_vllm}" == "true" ]; then
  if [ ! -f "${save_path}/config.json" ];then
    echo "moving config.json ..."
    cp "${proposer_base_path}/config.json" "${ckpt_path}/"
  fi
  python utils/convert_pt_file.py --ckpt_dir ${ckpt_path}

else
  ckpt_path=${ckpt_path}/LATEST/policy.pt
fi


nt=-1  # change to 5 for more samples
gpu=0
log_file=l0.log
top_p=0.9

nohup python infer.py \
  --config-path=${ckpt_path} \
  mode=sample_inst \
  exp_name=${exp_name} \
  ++use_gpus=${gpu} \
  ++use_vllm=${use_vllm} \
  ++n_samples=full \
  ++num_try=${nt} \
  ++model.eval_batch_size=16 \
  ++model.max_new_length=512 \
  ++top_p=${top_p} \
  ++temperature=1.0 \
  ++torch_type=float32 \
  ++samples_dir=samples/ \
  ++sample_file_name=${sample_file_mark} \
  ++test_file=${test_file} \
  ++start_idx=${start_idx} \
  ++end_idx=${end_idx} \
  ++target_key="predict_instruction" \
  ++model.name_or_path=${proposer_base_path} \
  ++local_run_dir=${ckpt_path} \
  ++saved_policy=${ckpt_path} > ${log_file} 2>&1 &
