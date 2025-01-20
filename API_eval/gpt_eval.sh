#!/bin/bash

eval_file=$1
exp_name=$2

python run_parallel.py \
  --input_file_path ${eval_file} \
  --exp_name ${exp_name} \
  --num_threads 30 \
  --model gpt-4o-mini \
  --temperature 0

