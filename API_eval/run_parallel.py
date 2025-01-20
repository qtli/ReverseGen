import json
import os
import argparse
import sys
import openai
from openai import OpenAI

import time
import random
import threading

project_path = os.path.join(os.getcwd().split("ReverseGen")[0], "ReverseGen")  # The path of the currently invoked outermost script
sys.path.append(project_path)
from utils.common_prompt import evaluate_mmlu, evaluate_math


API_KEY = ""
BASE_URL = ""

def formulate_eval_prompt(prompt, task):
    if "honesty" in task:
        subject = prompt.split(" about")[1].split(":\n\n")[0].strip()
        eval_p = evaluate_mmlu(subject=subject, question=prompt)


    elif "math" in task:
        eval_p = evaluate_math(question=prompt)

    else:
        print("Specify your own evaluation prompt format !!")
        eval_p = prompt

    return eval_p


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    return data


def write_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')



def filter_prompts(task, in_file_path, out_file_path):
    in_data = json.load(open(in_file_path))

    if os.path.exists(out_file_path):
        f = open(out_file_path, 'r', encoding="utf-8")
        already_generated_prompts = []
        for idx, item in enumerate(f):
            print(idx)
            try:
                already_generated_prompts.append(json.loads(item)['prompt'])
            except json.decoder.JSONDecodeError:
                continue

        temp_data = []
        for item in in_data:
            p = item["predict_instruction"]

            if isinstance(p, str):
                if p not in already_generated_prompts:
                    wrap_p = formulate_eval_prompt(p, task)
                    item["eval_prompt"] = wrap_p
                    temp_data.append(item)
            else:
                exit()
        return temp_data

    else:
        unique_prompts = []
        temp_data = []
        for item in in_data:
            p = item["predict_instruction"]
            if isinstance(p, str):
                if p not in unique_prompts:
                    unique_prompts.append(p)
                    wrap_p = formulate_eval_prompt(p, task)
                    item["eval_prompt"] = wrap_p
                    temp_data.append(item)
            else:
                exit()

        return temp_data


def process_data(max_tokens, data_chunk, output_file_path, model, temperature, top_p, trials=10, sys_mes=""):
    client = OpenAI(api_key='<KEY>')
    client.api_key = API_KEY
    client.base_url = BASE_URL


    for ix, row in enumerate(data_chunk):
        msg = [
            {"role": "system", "content": sys_mes},
            {"role": "user", "content": row["eval_prompt"] + " "},
        ]
        success = False
        for attempt in range(trials):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=msg,
                    temperature=temperature,
                    top_p=1,
                    max_tokens=max_tokens,  # The maximum number of [tokens](/tokenizer) that can be generated in the chat completion.
                )
            except Exception as e:
                time.sleep(random.randint(3, 30))
                print(f" Occurred: {e}. Retrying...")
            except openai.error.APIerror:
                time.sleep(random.randint(3, 30))
                print(f"APIError")
            else:
                success = True
                break
        if success:
            response = response.choices[0].message.content
            row["api_response"] = response
            row['response_source'] = model
            line = json.dumps(row, ensure_ascii=False)
            with open(output_file_path, 'a+', encoding='utf-8') as f:
                f.write(line + '\n')
            if (ix+1)%10==0:
                print(f"{ix+1} data finished")



def run_api(args, sys_mes="You are a helpful assistant."):
    # Assuming 'trials' is defined somewhere in the code
    trials = 10
    num_threads = 20

    input_file_path = args.input_file
    output_file_path = args.output_file
    model = args.model_name
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.model.max_new_length
    task = args.exp_name

    input_data = filter_prompts(task,
                                input_file_path,
                                output_file_path)
    print(f"========= {len(input_data)} prompts ============== prompt 0 ================")
    print(input_data[0]["eval_prompt"])

    # Split data into chunks for threading
    chunk_size = len(input_data) // num_threads
    data_chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]

    threads = []
    for data_chunk in data_chunks:
        thread = threading.Thread(target=process_data,
                                  args=(max_tokens, data_chunk, output_file_path, model, temperature, top_p, trials, sys_mes))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All threads completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Openai api")
    parser.add_argument("--input_file", type=str, help="input file name with suffix")
    parser.add_argument("--output_file", type=str, help="input file name with suffix")
    parser.add_argument("--exp_name", type=str, default="", help="honesty, math, ...")
    parser.add_argument("--model.max_new_length", type=str, default="gpt-4o-mini", help="gpt-4-1106-preview, gpt-4-0613, ...")
    parser.add_argument("--temperature", type=float, default=0, help="temperature")
    parser.add_argument("--top_p", type=float, default=1, help="top_p")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_threads", type=int, default=20, help="Define the number of threads")
    parser.add_argument("--sys_mes", type=str, default="cot")

    args = parser.parse_args()
    run_api(args)