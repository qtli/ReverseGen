#!/bin/bash
conda activate reversegen

set -e
TIME=$(date "+%m%d-%H%M%S") && echo -e "--------------------\nTime: $TIME\n--------------------"

GPU_ID=0
emb_size=4096
num_clusters=5
niter=1000
nredo=5

model_name_or_path=$1
eval_file=$2

dataset_size=$(python get_data_size.py ${eval_file})
echo "dataset_size: ${dataset_size}"

run_name=$3
echo "run_name: ${run_name}"

output_dir=./outputs/${run_name}
emb_memory_loc=${output_dir}/embeddings.npy
paths_memory_loc=${output_dir}/paths.npy
sorted_clusters_file_loc=${output_dir}/sorted_clusters
semdedup_pruning_tables_path=${output_dir}/dataframes

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

steps=(1 2 3 4 5)


if [[ " ${steps[@]} " =~ " 1 " ]]; then
  echo "Step 1: Generating embeddings..."
  python get_embeddings.py \
      --model_name_or_path ${model_name_or_path} \
      --padding_side "left" \
      --truncation_side "left" \
      --use_flash_attn "False" \
      --eval_file ${eval_file} \
      --overwrite_cache "False" \
      --preprocessing_num_workers 32 \
      --max_seq_length 256 \
      --seed 42 \
      --data_seed 42 \
      --report_to "wandb" \
      --output_dir ${output_dir} \
      --overwrite_output_dir "False" \
      --prediction_loss_only "False" \
      --per_device_eval_batch_size 1 \
      --remove_unused_columns "True" \
      --logging_steps 10 \
      --group_by_length "True" \
      --length_column_name "length" \
      --logging_strategy "steps" \
      --torch_dtype "float16" \
      --bf16 "False" \
      --fp16 "True"
fi


if [[ " ${steps[@]} " =~ " 2 " ]]; then
  echo "Step 2: Computing centroids..."
  python compute_centroids.py \
      --sim_metric "cosine" \
      --keep_hard True \
      --Kmeans_with_cos_dist True \
      --emb_memory_loc ${emb_memory_loc} \
      --sorted_clusters_file_loc ${sorted_clusters_file_loc} \
      --save_folder ${output_dir} \
      --ncentroids ${num_clusters} \
      --dataset_size ${dataset_size} \
      --emb_size ${emb_size} \
      --niter ${niter} \
      --nredo ${nredo} \
      --seed 42 \
      --path_str_dtype "U10000"
  # --text_emb_memory_loc  None  \
  # --paths_memory_loc  "<your path>.npy"  \
fi


if [[ " ${steps[@]} " =~ " 3 " ]]; then
  echo "Step 3: Sorting clusters..."
  python sort_clusters.py \
      --sim_metric "cosine" \
      --keep_hard True \
      --Kmeans_with_cos_dist True \
      --emb_memory_loc ${emb_memory_loc} \
      --paths_memory_loc ${paths_memory_loc} \
      --sorted_clusters_file_loc ${sorted_clusters_file_loc} \
      --save_folder ${output_dir} \
      --ncentroids ${num_clusters} \
      --dataset_size ${dataset_size} \
      --emb_size ${emb_size} \
      --niter ${niter} \
      --seed 42 \
      --path_str_dtype "U10000"
  # --text_emb_memory_loc  None  \
  # --paths_memory_loc  "<your path>.npy"  \
fi


#eps_values=(0.00095 0.03 0.04 0.05 0.06 0.07)
eps_values=(0.04)


if [[ " ${steps[@]} " =~ " 4 " ]]; then
  echo "Step 4: Running advanced SemDedup..."
  if [ -d ${semdedup_pruning_tables_path} ]; then
    rm -rf ${semdedup_pruning_tables_path}
  fi

  python advance_semdedup.py \
      --sim_metric "cosine" \
      --keep_hard True \
      --Kmeans_with_cos_dist True \
      --emb_memory_loc ${emb_memory_loc} \
      --paths_memory_loc ${paths_memory_loc} \
      --sorted_clusters_file_loc ${sorted_clusters_file_loc} \
      --save_folder ${output_dir} \
      --ncentroids ${num_clusters} \
      --dataset_size ${dataset_size} \
      --emb_size ${emb_size} \
      --niter ${niter} \
      --eps_list ${eps_values[*]} \
      --seed 42 \
      --largest_cluster_size_to_process 10000000 \
      --which_to_keep "hard" \
      --path_str_dtype "U10000"
  # --text_emb_memory_loc  None  \
  # --paths_memory_loc  "<your path>.npy"  \

  #  --eps_list 0.00095 0.03 0.04 0.05 0.06 0.07 \

  fi



if [[ " ${steps[@]} " =~ " 5 " ]]; then
  echo "Step 5: Extracting deduped data..."
  for eps in "${eps_values[@]}"; do
      output_txt_path=${output_dir}/output_eps${eps}.txt

      python extract_dedup_data.py \
          --output_txt_path ${output_txt_path} \
          --semdedup_pruning_tables_path ${semdedup_pruning_tables_path} \
          --sorted_clusters_path ${sorted_clusters_file_loc} \
          --eps ${eps} \
          --num_clusters ${num_clusters} \
          --raw_file_path ${eval_file}
  done
fi