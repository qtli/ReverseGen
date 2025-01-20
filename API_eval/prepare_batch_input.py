import json
import pdb
from run_parallel import process_batch_data, one_time_infer


def prepare_input(
    input_file="../samples/llama7b_as_explore_gsm8k_sft_only_questions/llama7b_as_explore_gsm8k_sft_only_questions_top_0.9_temp_1.0_st_5_test_raw_for_dpo_s0e10000-corrected.json",
    output_file="../samples/llama7b_as_explore_gsm8k_sft_only_questions/llama7b_as_explore_gsm8k_sft_only_questions_top_0.9_temp_1.0_st_5_test_raw_for_dpo_s0e10000-corrected-for-api-s0e5000.json",
):
    input_data = json.load(open(input_file))

    print(len(input_data))

    small_input_data = input_data[:100]

    batch_input_file = open(output_file, "w")
    for idx, item in enumerate(small_input_data):
        format_item = {"custom_id": f"request-{idx}",
                       "method": "POST",
                       "url": "/v1/chat/completions",
                       "body": {"model": "gpt-4o-mini",
                                "messages": [{"role": "system",
                                              "content": ""},  # You are a helpful assistant.
                                             {"role": "user",
                                              "content": item["formatted_question"]}],
                                "max_tokens": 512}}
        batch_input_file.write(json.dumps(format_item) + "\n")




input_file="../samples/llama7b_as_explore_gsm8k_sft_only_questions/llama7b_as_explore_gsm8k_sft_only_questions_top_0.9_temp_1.0_st_5_test_raw_for_dpo_s0e10000-format-for-api.json"
output_file="../samples/llama7b_as_explore_gsm8k_sft_only_questions/llama7b_as_explore_gsm8k_sft_only_questions_top_0.9_temp_1.0_st_5_test_raw_for_dpo_s0e10000-format-for-batch-api-s0e5000.jsonl"
prepare_input(
    input_file=input_file,
    output_file=output_file,
)
process_batch_data(batch_input_file=output_file)
# process_batch_data(batch_input_file="../samples/llama7b_as_explore_gsm8k_sft_only_questions/test.jsonl")


