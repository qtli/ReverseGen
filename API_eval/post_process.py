import pdb
import openai
from openai import OpenAI
import os
import re
import json
from tqdm import tqdm
from collections import Counter
import sys
import os

# from .run_parallel import one_time_infer, delete_extra_zero

from infer_answer import extract_ans_from_response, parse_pred
def test_api1():
    client = OpenAI(api_key='<KEY>')
    # client.api_key = "sk-145001708e4878be79699853996d50a0541d774cc6958b18473e9c56912d544b"
    # client.base_url = 'https://api.chatnio.net/v1'
    client.api_key = "sk-JhYefCHUrBp9zEI2B575E896AeC64f0c92042f2492Cb35Ae"
    client.base_url = "https://api.gpts.vin/v1"
    msg = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "你是谁开发的"},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        # model="gpt-4-turbo-preview",
        # model="gpt-4-turbo-1106",
        # model="gpt-4-0613",
        messages=msg,
        temperature=0.7,
        top_p=1,
        max_tokens=1024
    )
    response = response.choices[0].message.content
    print(response)

def test_api2():
    client = OpenAI(
        # This is the default and can be omitted
        api_key="sk-JhYefCHUrBp9zEI2B575E896AeC64f0c92042f2492Cb35Ae",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )
    response = chat_completion.choices[0].message.content
    print(response)


def check_consistency_between_turbo_attack_model(input_file, output_file, datasource="mmlu"):
    f = open(input_file, 'r', encoding="utf-8")
    if os.path.exists(output_file):
        # 调整后的文件，调整过的数据，保存在previous_data里
        previous_data = []
        for idx, line in enumerate(open(output_file, "r", encoding="utf-8")):
            try:
                p = json.loads(line)["prompt"]
                previous_data.append(p)
            except json.decoder.JSONDecodeError:
                print("----", idx)
    print(f"we have processed {len(previous_data)} prompts")

    wf = open(output_file, "a+", encoding="utf-8")
    inconsis = 0
    consis = 0
    invalid_eval = 0
    invalid_question = 0
    valid_question = 0
    invalid_format = 0

    invalid_patterns = ["Invalid"]
    valid_patterns = ["correct answer is", "Answer: "]
    unique_prompts = []
    for line_idx, line in enumerate(f):
        try:
            item = json.loads(line)
            if item["prompt"] not in unique_prompts:
                # raw文件里的重复数据不处理
                unique_prompts.append(item["prompt"])
            else:
                continue
        except json.decoder.JSONDecodeError:
            continue
        if item["prompt"] in previous_data:
            # 处理过的数据，不处理
            continue

        explore_model_pred = item["question"].split("Answer:")[1].strip(".").strip()
        api_pred = item["api_response"]

        print(line_idx)
        if api_pred in invalid_patterns:
            print("invoke gpt-4 ...")
            api_pred = one_time_infer(prompt=item["formatted_question"],
                                      model="gpt-4-0613")
            item["gpt4_api_response"] = api_pred
            print("gpt4: ", api_pred)
        wf.write(json.dumps(item) + "\n")

        flag = False
        for vp in valid_patterns:
            if vp in api_pred:
                flag = True
        if flag:
            # 这个问题是合理的
            if "Answer:" not in api_pred:
                # print(api_pred)
                invalid_eval += 1
                continue
            api_pred = api_pred.split("Answer:")[1].strip()[0]
            if api_pred not in "ABCDE":
                # print(item["api_response"])
                invalid_format += 1
                # print("====================================")
                continue

            if explore_model_pred != api_pred:
                inconsis += 1
            else:
                consis += 1
            valid_question += 1

        else:
            invalid_question += 1

    print(f"answer consistency: {consis} ({consis/(consis+inconsis)}); inconsistency: {inconsis} ({inconsis/(consis+inconsis)})")
    print("invalid eval among valid questions: ", invalid_eval)
    print("valid questions: ", valid_question)
    print("invalid questions: ", invalid_question)
    print("invalid_format: ", invalid_format)


def revise_answer_labels(input_file, output_file, datasource="mmlu"):
    print(input_file)
    f = open(input_file, 'r', encoding="utf-8")
    outputs = []
    invalid_patterns = ["Invalid"]
    if datasource == "mmlu":
        valid_patterns = ["correct answer is", "Answer:", "Correct Answer Evaluation:", "the answer is", "The valid answer is:"]

    for line_idx, line in enumerate(f.readlines()):
        # print(line_idx)
        try:
            item = json.loads(line)
        except json.decoder.JSONDecodeError:
            continue

        if datasource == "mmlu":
            explore_model_pred_question = item["question"].split("Answer:")[0]
            api_pred_raw = item["api_response"] if "gpt4_api_response" not in item else item["gpt4_api_response"]

            if invalid_patterns[0].lower() not in api_pred_raw.lower():
                api_pred_filter = api_pred_raw.replace("Answer: Valid. ", "").replace("[Option ", "").lower()

                flag = False
                for vp in valid_patterns:
                    vp = vp.lower()
                    if vp in api_pred_filter:
                        api_pred = api_pred_filter.split(vp)[-1].strip()[0].upper()
                        flag = True
                        break

                if flag == False:
                    continue
                if api_pred not in "ABCDE":
                    continue

                explore_model_pred_question += ("Answer:"+api_pred+".")
                if "\nAnswer:" not in explore_model_pred_question:
                    explore_model_pred_question = explore_model_pred_question.replace("Answer:", "\nAnswer:")
        else:
            api_pred_raw = item["api_response"]
            if invalid_patterns[0].lower() in api_pred_raw.lower():
                # print(api_pred_raw)
                # pdb.set_trace()
                continue
            if "solution:" not in api_pred_raw.lower():
                print(api_pred_raw)
                continue
            solution = api_pred_raw.split("Solution:")[1].strip()
            pattern = "####(.*)"
            answer = ""
            if len(re.findall(pattern, solution)) >= 1:
                answer = re.findall(pattern, solution)[-1].strip(' ')

            answer = delete_extra_zero(answer)
            item["solution"] = solution
            item["answer"] = answer

        outputs.append(item)
    json.dump(outputs, open(output_file, "w"), indent=2)
    print("outputs: ", len(outputs))



def collect_valid_queries(input_file, datasource="gsm8k"):
    def extract_numbers_from_string(input_string):
        numbers_list = re.findall(r'\d+', input_string)
        return numbers_list

    input_file_suffix = input_file.split(".")[-1]
    output_file = ".".join(input_file.split(".")[:-1]) + "_valid." + input_file_suffix
    print("writing to ", output_file)

    valid_data = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
                # print(idx)
            except UnicodeDecodeError:
                continue
            except json.decoder.JSONDecodeError:
                continue
            if "invalid" in item["api_response"].lower():
                continue
            if "####" not in item["api_response"]:
                continue

            api_response_answer = item["api_response"].split("####")[1].strip()
            if api_response_answer == "0":
                continue
            number_count = len(extract_numbers_from_string(api_response_answer))
            if number_count > 1:
                continue

            valid_data.append(item)
    print(len(valid_data))
    json.dump(valid_data, open(output_file, "w"), indent=2)


def collect_evol_inst(input_file):
    def extract_numbers_from_string(input_string):
        numbers_list = re.findall(r'\d+', input_string)
        return numbers_list

    input_file_suffix = input_file.split(".")[-1]
    output_file = ".".join(input_file.split(".")[:-1]) + "_valid." + input_file_suffix
    print("writing to ", output_file)

    valid_data = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError")
                continue
            except json.decoder.JSONDecodeError:
                print("JSONDecodeError")
                continue
            if "####" not in item["api_response"]:
                print("#### error")
                print(item["api_response"])
                continue
            api_response_answer = item["api_response"].split("####")[1].strip()
            number_count = len(extract_numbers_from_string(api_response_answer))
            if number_count > 1:
                print("answer error")
                continue
            del item["question"]
            del item["answer"]
            item["question"] = item["formatted_question"]
            item["model_prediction_gpt-4o-mini"] = item["api_response"]
            del item["api_response"]
            del item["formatted_question"]
            valid_data.append(item)
    print(len(valid_data))
    json.dump(valid_data, open(output_file, "w"), indent=2)



def combine_multiple_answers(sample_files, output_file):
    total = 0
    disagree = 0

    question_to_items = []
    for file in sample_files:
        s1 = open(file, encoding="utf-8")
        question_to_item = {}
        for sidx, line in enumerate(s1):
            item = json.loads(line)
            question_to_item[item["question"]] = item
        question_to_items.append(question_to_item)


    new_data = []
    for idx, question in enumerate(question_to_items[0]):
        greedy_item = question_to_items[0][question]
        try:
            init_answer = parse_pred(prediction=greedy_item["api_response"])
        except IndexError:
            print("INIT IndexError: ======", greedy_item["api_response"])
            continue

        samples_answers = []
        samples_solutions = []
        for q2i in question_to_items:
            item = q2i[question]
            try:
                sample_answer = parse_pred(prediction=item["prediction"][0])
            except IndexError:
                print("SAMPLE IndexError: ======", item["prediction"][0])
                continue
            if sample_answer == "none":
                continue
            samples_answers.append(sample_answer)
            samples_solutions.append(item["prediction"][0])

        samples_answers.append(init_answer)
        samples_solutions.append(greedy_item["api_response"])
        pred_freq = Counter(samples_answers)
        pred = sorted(pred_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
        pred_num = sorted(pred_freq.items(), key=lambda x: x[1], reverse=True)[0][1]

        if pred_num == 1:
            disagree += 1
            continue
        # if len(samples_answers) == 1:
        #     continue
        # else:
        #     disagree +=11

        total += 1
        greedy_item["answer"] = pred
        pred_idx = samples_answers.index(pred)
        greedy_item["model_prediction_gpt-4o-mini"] = samples_solutions[pred_idx].strip(".").strip()
        del greedy_item["prediction"]
        new_data.append(greedy_item)

    print("final: ", len(new_data))
    print("disagree: ", disagree)
    json.dump(new_data, open(output_file, "w"), indent=2)


if __name__ == '__main__':
    # test_api1()
    # test_api1()

    # check_consistency_between_turbo_attack_model(input_file="output/vicuna-7b-on-sft-mmlu-3907-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json")
    # check_consistency_between_turbo_attack_model(input_file="output/llama7b-sft-mmlu-2448-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json")
    # answer consistency: 838 (0.382648401826484); inconsistency: 1352 (0.617351598173516)
    # invalid eval among valid questions:  100
    # valid questions:  2190
    # invalid questions:  1461
    # invalid_format:  232

    # check_consistency_between_turbo_attack_model(input_file="output/vicuna-7b-on-dpo-mmlu-3920-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json")
    # answer consistency: 418 (0.218848167539267); inconsistency: 1492 (0.7811518324607329)
    # invalid eval among valid questions:  142
    # valid questions:  1910
    # invalid questions:  1598
    # invalid_format:  249

    # check_consistency_between_turbo_attack_model(input_file="output/vicuna_response_llama7b_mmlu_dpo_victim_vicuna_3332_4744_questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json",
    #                                              output_file="output/vicuna_response_llama7b_mmlu_dpo_victim_vicuna_3332_4744_questions_gpt-3.5-turbo-1106_output_t0.0_tp1-adjust.json")

    # check_consistency_between_turbo_attack_model(input_file="output/vicuna_response_llama7b_mmlu_sft_2448_eval_sft_MMLU_2_all-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json",
    #                                              output_file="output/vicuna_response_llama7b_mmlu_sft_2448_eval_sft_MMLU_2_all-questions_gpt-3.5-turbo-1106_output_t0.0_tp1-adjust.json")

    # revise_answer_labels(input_file="output/vicuna-7b-on-dpo-mmlu-3920-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json",
    #                      output_file="output/vicuna-7b-on-dpo-mmlu-3920-questions-after-turbo.json")


    # revise_answer_labels(input_file="output/llama7b-sft-mmlu-2448-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json",
    #                      output_file="output/llama7b-sft-mmlu-2448-questions-after-turbo.json")


    # revise_answer_labels(input_file="output/vicuna_response_llama7b_mmlu_dpo_victim_vicuna_3332_eval_1_0_1000-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json",
    #                      output_file="output/llama7b_mmlu_dpo_victim_vicuna_3332_eval_1_0_1000-questions-after-turbo.json")

    # revise_answer_labels(
    #     input_file="output/vicuna_response_llama7b_mmlu_sft_2448_eval_sft_MMLU_2_all-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json",
    #     output_file="output/llama7b_mmlu_sft_2448_eval_sft_MMLU_2_all-questions-after-turbo.json")

    # revise_answer_labels(
    #     input_file="output/old/llama7b_as_explore_gsm8k_sft_only_questions_top_0.9_temp_1.0_st_5_test_raw_for_dpo_s0e10000-format-for-api_gpt-4o-mini_output_t0.0_tp1.json",
    #     output_file="output/gsm8k/llama7b_as_explore_gsm8k_sft_only_questions_top_0.9_temp_1.0_st_5_test_raw_for_dpo_s0e10000-format-for-api_gpt-4o-mini_output_t0.0_tp1-after-gpt4omini.json",
    #     datasource="gsm8k"
    # )

    # collect_valid_queries(
    #     input_file="output/llama3_8b_as_explore_math_sft_top_0.98_temp_1.0_st_5_raw_train_dpo_vllm_s0e10000-format-for-api_gpt-4o-mini_output_t0.0_tp1.json",
    #     output_file="output/llama3_8b_as_explore_math_sft_top_0.98_temp_1.0_st_5_raw_train_dpo_vllm_s0e10000-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid.json",
    #     datasource="gsm8k"
    # )


    # collect_valid_queries(
    #     input_file="output/llama3_8b_as_explore_gsm_sft_0924_victim_llama3_8b_instruct_dpo1_balance_human_top_0.9_temp_1.0_st_5_raw_train_dpo_vllm_s10000e20000-format-for-api_gpt-4o-mini_output_t0.0_tp1.json",
    #     datasource="gsm8k"
    # )

    # collect_evol_inst(
    #     input_file="output/evol_inst/test_on_evol_inst-format-for-api_gpt-4o-mini_output_t0.0_tp1.json"
    # )

    # combine_multiple_answers(
    #     sample_files=[
    #         "output/llama3_8b_as_explore_gsm_sft_victim_llama3_8b_instruct_dpo1_top_0.9_temp_1.0_st_1_test_vllm-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_more_samples.json",
    #         "output/llama3_8b_as_explore_gsm_sft_victim_llama3_8b_instruct_dpo1_top_0.9_temp_1.0_st_1_test_vllm-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_more_samples_greedy.json",
    #
    #     ],
    #     output_file="output/llama3_8b_as_explore_gsm_sft_victim_llama3_8b_instruct_dpo1_top_0.9_temp_1.0_st_1_test_vllm-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_with_answer.json"
    # )
    # final:  108
    # disagree:  137

    # final: 159
    # disagree: 86

    # combine_multiple_answers(
    #     sample_files=[
    #         "output/llama3_8b_as_explore_gsm_sft_0924_victim_llama3_8b_instruct_dpo1_balance_human_top_0.9_temp_1.0_st_5_raw_train_dpo_vllm_s10000e20000-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_more_samples.json",
    #     ],
    #     output_file="output/llama3_8b_as_explore_gsm_sft_0924_victim_llama3_8b_instruct_dpo1_balance_human_top_0.9_temp_1.0_st_5_raw_train_dpo_vllm_s10000e20000-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_with_answer.json"
    # )

    # f = open("output/vicuna_response_llama7b_mmlu_sft_2448_eval_sft_MMLU_2_all-questions_gpt-3.5-turbo-1106_output_t0.0_tp1.json", "r", encoding="utf-8")
    #
    # count = 0
    # unique_prompts = []
    # for line in f:
    #     try:
    #         api_resp = json.loads(line)["api_response"]
    #         if json.loads(line)["prompt"] not in unique_prompts:
    #             unique_prompts.append(json.loads(line)["prompt"])
    #     except json.decoder.JSONDecodeError:
    #         continue
    #     if api_resp.strip() == "Invalid":
    #         count += 1
    # print(count)

    # emb = one_time_infer(
    #     prompt="i think the food is good",
    #     model="text-embedding-3-small"
    # )
    #

    # file = open("output/prompt_4o_with_diff_data_gpt4_eval_output.json", "r", encoding="utf-8")
    # data = []
    # for line in file.readlines():
    #     try:
    #         item = json.loads(line)
    #         new_item = {
    #             "prompt": item["inst_prompt"],
    #             "question": item["prompt"],
    #             "solution": item["api_response"]
    #         }
    #         data.append(new_item)
    #     except json.decoder.JSONDecodeError:
    #         pass
    # json.dump(data, open("output/prompt_4o_with_diff_data_with_answer.json", "w"), indent=2)

    # file = json.load(open("output/prompt_4o_with_diff_data.json"))
    # avg_len_4 = 0
    # demo_len = 0
    # c = 0
    # for item in file:
    #     p = item["prompt"]
    #     demons = item["inst_prompt"].replace("Ask math questions:\n\n", "").replace("\n\nQ4.", "").strip()
    #     ds = re.split(r'\s*Q[1-3]\.\s*', demons)
    #     for d in ds:
    #         if d.strip() != 0:
    #             demo_len += len(d.split())
    #             c += 1
    #     avg_len_4 += len(p.split())
    #     pdb.set_trace()
    #
    # avg_len_4 = avg_len_4 / len(file)
    # print(avg_len_4)
    # print(demo_len/c)


    file = json.load(open("../explore_dataset/llama3_8b_as_explore_gsm_sft_victim_llama3_8b_instruct_dpo1/aaa.json"))
    avg_len = 0
    for item in file:
        p = item["question"]
        avg_len += len(p.split())
    print(avg_len/len(file))