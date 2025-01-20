#!/bin/sh
base_path=$1
step=$2
exp_name=$3
train_file=$4
test_file=$5
prev_ckpt_path=$6

ckpt_path=/cache/ckpt/reversegen
use_gpus=\'0,1,2,3,4,5,6,7\'


echo  "=================== Exp_name: ${exp_name}; Proposer model base: ${proposer_name} ==================="

if [ "$step" == "warmup" ]; then
  echo "conduct warmup sft training ... train file: ${train_file}"
  echo "exp_name: ${exp_name}"
  nohup python train.py \
    loss=sft \
    model=${model_name} \
    datasets=[warmup_data] \
    exp_name=${exp_name} \
    mode=train \
    enable_lora=${enable_lora} \
    train_file=${train_file} \
    ++test_file=${test_file} \
    ++model.name_or_path=${base_path} \
    ++loss.dataloader=SFTDataLoader \
    ++cache_dir=${ckpt_path} \
    ++wandb.enabled=True \
    model.batch_size=8 > ${exp_name}.log 2>&1 &

elif [ "$step" == "preference" ]; then
  echo "conduct the first preference learning ..."
  nohup python train.py \
    loss=dpo \
    model=${model_name} \
    datasets=[dpo_data] \
    ++loss.dataloader=PairedPreferenceDataLoader \
    exp_name=${exp_name} \
    mode=train \
    lr=5e-5 \
    enable_lora=${enable_lora} \
    ++train_file=${train_file} \
    ++test_file=${test_file} \
    ++cache_dir=${ckpt_path} \
    ++model.load_from=${prev_ckpt_path}/LATEST/policy.pt \
    ++model.name_or_path=${base_path} \
    model.batch_size=8 > ${exp_name}".log" 2>&1 &


else
  echo "conduct sft training ... on explored train file: ${train_file}"
  echo "exp_name: ${exp_name}"
  nohup python train.py \
    loss=sft \
    model=${model_name} \
    datasets=[explored_data] \
    exp_name=${exp_name} \
    mode=train \
    train_file=${train_file} \
    ++test_file=${test_file} \
    ++model.name_or_path=${base_path} \
    ++loss.dataloader=SFTDataLoader \
    ++cache_dir=${ckpt_path} \
    ++wandb.enabled=True \
    lr=5e-5 \
    enable_lora=${enable_lora} \
    model.max_length=2048 \
    ++prompt_key=${prompt_key} \
    ++target_key=${target_key} \
    model.batch_size=8 > ${exp_name}.log 2>&1 &

fi

