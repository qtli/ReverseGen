import json
import pdb
import os
import threading
import re
from collections import Counter
from openai import OpenAI
# from run_parallel import cot_prompt_map_func, delete_extra_zero


gsm8k_nshots = [
    (
        'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
        'There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = <<21-15=6>>6 trees that were planted.\n#### 6'
    ),
    (
        'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
        'There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = <<3+2=5>>5 cars are in the parking lot.\n#### 5'
    ),
    (
        'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
        'Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39 pieces left in total.\n#### 39'
    ),
    (
        'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
        'Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8 lollipops.\n#### 8'
    ),
    (
        'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
        'Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = <<2*2=4>>4 more toys. Now he has 5 + 4 = <<5+4=9>>9 toys.\n#### 9'
    ),
    (
        'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
        'There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = <<4*5=20>>20 computers were added. Now 9 + 20 = <<9+20=29>>29 computers are now in the server room.\n#### 29'
    ),
    (
        'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
        'Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = <<58-23=35>>35 at the end of Tuesday, and 35 - 2 = <<35-2=33>>33 at the end of wednesday.\n#### 33'
    ),
    (
        'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
        'Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = <<5*3=15>>15 dollars. Now she has 23 - 15 = <<23-15=8>>8 dollars left.\n#### 8'
    )
]

def nshot_chats(n: int, question: str) -> dict:
    # https://medium.com/@sewoong.lee/how-to-reproduce-llama-3s-performance-on-gsm-8k-e0dce7fe9926
    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f"Answer:\nLet's think step by step.\n{s}"

    chats = [
        {"role": "system", "content": "Your task is to solve a series of math word problems by providing the final answer. Use the format #### [value] to highlight your answer. Do not use latex code in your answer. Follow the answering format of the history answer. Make sure to carefully read and understand each problem before providing your answer."}
    ]

    for q, a in gsm8k_nshots[:n]:
        chats.append(
            {"role": "user", "content": question_prompt(q)})
        chats.append(
            {"role": "assistant", "content": answer_prompt(a)})

    # chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})
    chats.append({"role": "user", "content": question_prompt(question)})
    return chats


# client.api_key = 'sk-kBHtFHLJziDqAxSgEaA69e2a347f4c41B20035Fc3cB998D1'
# client.base_url = "https://api.gpts.vin/v1"

client = OpenAI(api_key='<KEY>')
# client.api_key = "sk-145001708e4878be79699853996d50a0541d774cc6958b18473e9c56912d544b"
client.api_key = "sk-lqoKaKhOty73BdDGE09f642a95Ca4a38879227566cBdB964"
client.base_url = 'https://api.gptplus5.com/v1'
# client.api_key = 'sk-hMPJOiETW51ema2u66C479E11cC9404dAc0bCdD45611Fb28'
# client.base_url = "https://api.gpts.vin/v1"

total_token_records = 0

def infer(samples, output_path="", st=1, model_name="gpt-4o-mini", temperature=0):
    global total_token_records
    wf = open(output_path, 'a+', encoding='utf-8')
    for si, s in enumerate(samples):
        question = s["question"]
        # prompt, inst = cot_prompt_map_func(question)
        # msg = [
        #     {"role": "system", "content": inst},
        #     {"role": "user", "content": prompt},  # + '\nLet\'s think step by step\n'
        #     {"role": "assistant", "content": prompt},  # + '\nLet\'s think step by step\n'
        #
        #     {"role": "user", "content": prompt},  # + '\nLet\'s think step by step\n'
        # ]

        msg = nshot_chats(n=3, question=question)
        responses = []
        for _ in range(st):
            response = client.chat.completions.create(
                model=model_name,
                messages=msg,
                temperature=temperature,
                max_tokens=512
            )
            total_token_records += response.usage.total_tokens

            response = response.choices[0].message.content
            responses.append(response)
            print(responses)

        s["prediction"] = responses
        s["prediction_source"] = model_name
        line = json.dumps(s, ensure_ascii=False)
        wf.write(line + '\n')
        if (si + 1) % 10 == 0:
            print(f"{si + 1} data finished")


def get_model_answer(model_name="gpt-4-0613", st=1, input_file="../explore_dataset/raw_data/gsm8k-format-data.json", temperature=0.0):
    gsm8k_test = json.load(open(input_file))
    # if output_file == "":
    #     output_file = f"test_gsm8k_test/{model_name}-on-gsm8k-test-st-{st}-raw.json"
    input_file_name = os.path.split(input_file)[1]
    input_file_dir = os.path.split(input_file)[0]
    ipt_file_suffix = input_file_name.split('.')[-1]
    output_file_name = ".".join(input_file_name.split('.')[:-1]) + "_more_samples."  + ipt_file_suffix
    output_file = os.path.join(input_file_dir, output_file_name)
    print("writing to ", output_file)

    print(output_file)
    if os.path.exists(output_file):
        previous_prompts = []
        for line in open(output_file, encoding="utf-8"):
            question = json.loads(line)["question"].strip()
            previous_prompts.append(question)

        new_prompts = []
        for item in gsm8k_test:
            if item["question"].strip() not in previous_prompts:
                new_prompts.append(item)
        print(f"in total we have {len(gsm8k_test)}, we have generated {len(previous_prompts)}, we only needs to process {len(new_prompts)}")

        gsm8k_test = new_prompts

    gsm8k_test = gsm8k_test
    num_threads = 20
    # Split data into chunks for threading
    chunk_size = len(gsm8k_test) // num_threads
    data_chunks = [gsm8k_test[i:i + chunk_size] for i in range(0, len(gsm8k_test), chunk_size)]

    threads = []
    for data_chunk in data_chunks:
        thread = threading.Thread(target=infer,
                                  args=(data_chunk, output_file, st, model_name, temperature))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All threads completed.")




def parse_pred(pattern="####(.*)", prediction=""):
    preds = re.findall(pattern, prediction)

    if len(preds) < 1:
        return "none"

    pred = delete_extra_zero(preds[-1].strip(" ")) if len(preds) >= 1 and bool(re.search(r"\d", preds[-1])) else ""

    if pred == "":
        pred = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', prediction)
        if len(pred) >= 1:
            pred = delete_extra_zero(pred[-1].replace(",", "").strip(".").strip(" "))
        else:
            pred = ""
    else:
        pred = delete_extra_zero(
            re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred.replace(",", ""))[0].strip(".").strip(" "))
    return pred


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer


def test_acc(output_path="gpt-4o-mini-on-gsm8k-test.json", st=1, writen_file="testmini_gpt4_eval.json"):
    print("majority voting over: ", st)

    new_data = []
    pattern = "####(.*)"
    count = 0
    total_count = 0
    for idx, line in enumerate(open(output_path, "r", encoding="utf-8")):
        # print("idx: ", idx)
        item = json.loads(line)
        answer = item["answer"]
        ans = re.findall(pattern, answer)[0].strip()

        if st == 1:
            if isinstance(item["prediction"], list):
                prediction = item["prediction"][0]
            else:
                prediction = item["prediction"]
            pred = parse_pred(prediction)
        else:
            preds = []
            for prediction in item["prediction"][:st]:
                pred = parse_pred(prediction)
                preds.append(pred)
            pred_freq = Counter(preds)
            pred = sorted(pred_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
            item["pred_answer_list"] = preds

        if pred == "none":
            item["match"] = None
            new_data.append(item)
            continue

        # print("prediction: ", prediction)
        # print("pred: ", pred)
        # print("====================================")

        item["pred_answer"] = pred
        if pred == ans:
            item["match"] = True
            count += 1
        else:
            item["match"] = False
        new_data.append(item)
        total_count +=1

    # json.dump(new_data, open(writen_file, "w"), indent=2)
    print("acc : ", count/total_count)



if __name__ == '__main__':
    # get_model_answer(model_name="gpt-4o-mini", st=5)

    # get_model_answer(model_name="gpt-4o-mini", st=1)


    # get_model_answer(model_name="gpt-4-0613", st=1)

    # test_acc(
    #     output_path="test_gsm8k_test/gpt-4o-mini-on-gsm8k-test-st-5-raw.json",
    #     st=1,
    #     writen_file="test_gsm8k_test/gpt-4o-mini-on-gsm8k-test-st-5-eval.json"
    # )
    #
    # test_acc(
    #     output_path="gpt-4-0613-on-gsm8k-test-st-1-raw.json",
    #     st=5,
    #     writen_file="gpt-4-0613-on-gsm8k-test-st-1-eval.json"
    # )

    # test_acc(
    #     output_path="test_gsm8k_test/gpt-4o-mini-on-gsm8k-test-st-1-raw.json",
    #     st=1,
    #     writen_file="test_gsm8k_test/gpt-4o-mini-on-gsm8k-test-st-1-eval.json"
    # )


    # get_model_answer(model_name="gpt-4o-mini",
    #                  input_file="output/llama3_8b_as_explore_math_sft_top_0.98_temp_1.0_st_5_raw_train_dpo_vllm_s0e10000-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid.json",
    #                  output_file="output/llama3_8b_as_explore_math_sft_top_0.98_temp_1.0_st_5_raw_train_dpo_vllm_s0e10000-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_more_3_samples.json",
    #                  st=1,
    #                  temperature=0.7)



    # get_model_answer(model_name="gpt-4o-mini",
    #                  input_file="output/llama3_8b_as_explore_math_sft_victim_llama3_8b_instruct_dpo1_top_0.98_temp_1.0_st_1_test_vllm-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid.json",
    #                  output_file="output/llama3_8b_as_explore_math_sft_victim_llama3_8b_instruct_dpo1_top_0.98_temp_1.0_st_1_test_vllm-format-for-api_gpt-4o-mini_output_t0.0_tp1_valid_more_answers.json",
    #                  st=3,
    #                  temperature=0.7)



    get_model_answer(model_name="gpt-4o-mini",
                     input_file="output/llama3_8b_as_explore_gsm_sft_0924_victim_llama3_8b_instruct_dpo1_balance_human/filter_data.json",
                     st=1,
                     temperature=0)