#!/bin/bash
eval_file=$1

python extract_filtered_data.py \
--raw_file ${eval_file} \
--dedup_mode minhash \
--minhash_threshold 0.78
