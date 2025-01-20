import json
import random
import argparse
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['safety', 'honesty', 'math'], default='safety')
args = parser.parse_args()

def safe_template(samples):
    prompt = f'''Ask questions:
1. {samples[0]}
2. {samples[1]}
3. {samples[2]}
4.'''
    target = samples[3]
    return prompt, target


def MMLU_template(samples, type):
    prompt = f'''Ask multiple choice questions (with answers) about {type}:\n
Q1. {samples[0]}

Q2. {samples[1]}

Q3. {samples[2]}

Q4.'''
    target = samples[3]
    return prompt, target

def math_template(samples, type="only_question"):
    if type == "only_question":
        prompt = f'''Ask math questions:\n
Q1. {samples[0]["question"]}

Q2. {samples[1]["question"]}

Q3. {samples[2]["question"]}

Q4.'''
        target = samples[3]["question"]

    else:
        prompt = f'''Ask math questions with answers:\n
Q1. {samples[0]["question"]}
{samples[0]["answer"]}

Q2. {samples[1]["question"]}
{samples[0]["answer"]}

Q3. {samples[2]["question"]}
{samples[0]["answer"]}

Q4.'''
        target = samples[3]["question"] + "\n" + samples[3]["answer"]

    return prompt, target

def gsm8k_template(samples, type="only_question"):
    if type == "only_question":
        prompt = f'''Ask math questions:\n
Q1. {samples[0]["question"]}

Q2. {samples[1]["question"]}

Q3. {samples[2]["question"]}

Q4.'''
        target = samples[3]["question"]

    else:
        prompt = f'''Generate math questions with answers:\n
Q1. {samples[0]["question"]}
{samples[0]["answer"]}

Q2. {samples[1]["question"]}
{samples[0]["answer"]}

Q3. {samples[2]["question"]}
{samples[0]["answer"]}

Q4.'''
        target = samples[3]["question"] + "\n" + samples[3]["answer"]

    return prompt, target


def custom_template(samples):
    raise NotImplementedError

def format_safe_warmup_data(
        hh_rlhf_warmup_train_size=10000,
        hh_rlhf_warmup_test_size=1000,
        hh_rlhf_dpo_train_size=10000,
        toxic_warmup_train_size=1700,
        toxic_warmup_test_size=300,
        toxic_dpo_train_size=300,
):
    # Read from the original data and retain only the instruction portions.
    cai_harmless = load_dataset("raw_data/cai-conversation-harmless")
    field_names = ["train_sft", "test_sft"]
    split_names = ["train", "test"]
    hh_rlhf_data = {}

    for field, split in zip(field_names, split_names):
        field_data = cai_harmless["field"]["init_prompt"]
        random.shuffle(field_data)
        field_data = list(set(field_data))
        hh_rlhf_data[split] = field_data
        print(f"hh_rlhf {split}: {len(field_data)}")

    field_names = [["train", "dev"], ["test"]]
    split_names = ["train", "test"]
    toxic_chat_data = {}
    for fields, split in zip(field_names, split_names):
        field_data = []
        for field in fields:
            field_data += [json.loads(line)["reddit_thread"][0]["text"].replace("Title:", "").strip().replace("\n", "") for line in
          open(f"raw_data/ToxiChat/{field}.jsonl")]
        field_data = list(set(field_data))
        random.shuffle(field_data)
        toxic_chat_data[split] = field_data
        print(f"toxicchat {split}: ", len(field_data))

    json.dump(hh_rlhf_data, open("raw_data/hh-rlhf-format-data.json", "w"), indent=2)
    json.dump(toxic_chat_data, open("raw_data/toxicchat-format-data.json", "w"), indent=2)


    if hh_rlhf_warmup_train_size == "all" or hh_rlhf_warmup_train_size > len(hh_rlhf_data["train"]):
        hh_rlhf_warmup_train_size = len(hh_rlhf_data["train"])
    if toxic_warmup_train_size == "all" or toxic_warmup_train_size > len(toxic_chat_data["train"]):
        toxic_warmup_train_size = len(toxic_chat_data["train"])
    if hh_rlhf_warmup_test_size == "all" or hh_rlhf_warmup_test_size > len(hh_rlhf_data["test"]):
        hh_rlhf_warmup_test_size = len(hh_rlhf_data["test"])
    if toxic_warmup_test_size == "all" or toxic_warmup_test_size > len(toxic_chat_data["test"]):
        toxic_warmup_test_size = len(toxic_chat_data["test"])

    warmup_train_samples = []
    dpo_train_samples = []
    test_samples = []
    prompt_hist = []


    for data_name, data_size in zip([hh_rlhf_data, toxic_chat_data], [[hh_rlhf_warmup_train_size, hh_rlhf_dpo_train_size, hh_rlhf_warmup_test_size], [toxic_warmup_train_size, toxic_dpo_train_size, toxic_warmup_test_size]]):
        print(f"sample {data_size[0]} training samples for warmup ...")
        split = "train"
        for i in range(data_size[0]):
            target = data_name[split][i]
            other_samples = data_name[split][:i] + data_name[split][(i+1):]

            while True:
                demons = random.sample(other_samples, 3)
                demons.append(target)
                prompt, target = safe_template(demons)

                if prompt not in prompt_hist:
                    break

            prompt_hist.append(prompt)
            warmup_train_samples.append({
                "prompt": prompt,
                "target": target
            })

        print(f"sample {data_size[1]} training samples for dpo ...")
        for i in range(data_size[1]):

            while True:
                demons = random.sample(data_name[split], 4)
                prompt, target = safe_template(demons)
                if prompt not in prompt_hist:
                    break

            prompt_hist.append(prompt)
            dpo_train_samples.append({
                "prompt": prompt,
                "target": target
            })

        print(f"sample {data_size[2]} samples for test ...")
        split = "test"
        for i in range(data_size):
            target = data_name[split][i]
            other_samples = data_name[split][:i] + data_name[split][(i+1):]

            while True:
                demons = random.sample(other_samples, 3)
                demons.append(target)
                prompt, target = safe_template(demons)
                if prompt not in prompt_hist:
                    break

            prompt_hist.append(prompt)
            test_samples.append({
                "prompt": prompt,
                "target": target
            })

    json.dump(warmup_train_samples, open("safety_red_teaming/warmup_train.json", "w"), indent=2)
    json.dump(dpo_train_samples, open("safety_red_teaming/iterative_preference_train.json", "w"), indent=2)
    json.dump(test_samples, open("safety_red_teaming/test.json", "w"), indent=2)

def split_mmlu_by_question_category():
    data = json.load(open("raw_data/MMLU/MMLU_uncertain.json"))
    print(len(data["instances"]))  # 2448
    instances = data["instances"]

    new_data = {}
    for instance in instances:
        text = instance["text"]
        prompt = text.split("\n\n")[0].strip() + "\n\n"
        last_question = text.split("\n\n")[-1].strip()
        only_qa = \
        last_question.split("Are you sure you accurately answered the question based on your internal knowledge?")[
            0].strip()
        cate = prompt.strip().split(") about")[1].strip().rstrip(".")
        if cate not in new_data:
            new_data[f"{cate}"] = []
        new_data[cate].append(only_qa)

    print(f"{len(new_data.keys())} cates: ", new_data.keys())
    for k in new_data:
        print(f"{k}: {len(new_data[k])}")
    '''
    MMLU:
    28 cates
    dict_keys(['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics'])
    high school biology: 155
    college mathematics: 50
    high school macroeconomics: 195
    high school microeconomics: 119
    high school government and politics: 97
    elementary mathematics: 189
    college medicine: 87
    college biology: 72
    high school mathematics: 135
    formal logic: 63
    econometrics: 57
    clinical knowledge: 133
    electrical engineering: 73
    high school chemistry: 102
    abstract algebra: 50
    global facts: 50
    business ethics: 50
    astronomy: 76
    college computer science: 50
    anatomy: 68
    high school geography: 99
    conceptual physics: 118
    college chemistry: 50
    college physics: 51
    high school european history: 83
    high school computer science: 50
    computer security: 50
    high school physics: 76
    '''
    json.dump(new_data, open("honesty_calibration/MMLU_uncertain_by_cate.json", "w"))


def format_honesty_warmup_data(
        dpo_size_per_cate=1700,
        test_size_per_cate=50
):

    data = json.load(open("honesty_calibration/MMLU_uncertain_by_cate.json"))
    prompt_hist = []

    warmup_train_data = []
    for key in data:
        cate_data = data[key]
        for idx, qa in enumerate(cate_data):
            cate_data_tmp = cate_data[:idx] + cate_data[(idx+1):]

            while True:
                combines = random.sample(cate_data_tmp, 3)
                combines.append(qa)
                prompt, target = MMLU_template(combines, type=key)
                if prompt not in prompt_hist:
                    break

            item = {"prompt": prompt, "target": target}
            prompt_hist.append(prompt)
            warmup_train_data.append(item)

    random.shuffle(warmup_train_data)
    json.dump(warmup_train_data, open("honesty_calibration/warmup_train.json", "w"))


    output_files = ["iterative_preference_train", "test"]
    for size, opt_f in zip([dpo_size_per_cate, test_size_per_cate], output_files):
        split_data = []

        for key in data:
            cate_data = data[key]
            while len(split_data) < size:

                while True:
                    demons = random.sample(cate_data, 3)
                    demons.append("")
                    prompt, target = MMLU_template(demons, type=key)

                    if prompt not in prompt_hist:
                        break

                prompt_hist.append(prompt)
                item = {"prompt": prompt, "target": target}
                split_data.append(item)

        json.dump(split_data, open(f"honesty_calibration/{opt_f}.json", "w"), indent=4)


def format_math_warmup_data(
        setting="general", # difficult
):
    def gsmplus():
        test_file = open("raw_data/gsmplus_testmini.jsonl", encoding="utf-8")
        data = []
        for idx, line in enumerate(test_file.readlines()):
            item = json.loads(line)
            data.append(item)
        json.dump(data, open("raw_data/gsmplus-format-data.json", "w"), indent=2)

    def gsm8k():
        train_file = open("raw_data/grade-school-math-master/grade_school_math/data/train.jsonl", "r")
        test_file = open("raw_data/grade-school-math-master/grade_school_math/data/test.jsonl", "r")
        question_to_answer = {}

        train_data = []
        for line in train_file.readlines():
            train_data.append(json.loads(line))

        test_data = []
        for line in test_file.readlines():
            test_data.append(json.loads(line))

        for item in train_data + test_data:
            question = item["question"].strip()
            answer = item["answer"]
            question_to_answer[question] = answer

        gsm8k_data = {
            "train": train_data,
            "test": test_data
        }
        json.dump(gsm8k_data, open("raw_data/gsm8k-format-data.json", "w"), indent=2)
        json.dump(question_to_answer, open("raw_data/gsm8k-question-to-answer.json", "w"), indent=2)

    def math_step_dpo():
        dataset = load_dataset("raw_data/Math-Step-DPO-10K")["train"]
        train_data = []
        for item in dataset:
            train_data.append({
                "question": item["prompt"].strip(),
                "answer": item["full_chosen"],
                "answer_span": item["answer"],
                "source": item["dataset"]
            })
        json.dump(train_data, open("raw_data/math-step-dpo-format-data.json", "w"), indent=2)

    def mmiqa():
        gsm_question_hist = []
        math_question_hist = []
        with open("raw_data/mmiqc.jsonl", "r") as f:
            format_data = defaultdict(list)
            for line in tqdm(f.readlines()):
                item = json.loads(line)
                question = item["instruction"].strip().replace("Please solve the following problem and put your answer at the end with \"The answer is: \".", "").strip()

                if item["source"] in ["GSM_FOBAR", "GSM_Rephrased"]:
                    if question in gsm_question_hist:
                        continue
                    gsm_question_hist.append(question)

                # else:
                #     if question in math_question_hist:
                #         continue
                #     math_question_hist.append(question)

                format_data[item["source"]].append({
                    "question": question,
                    "answer": item["output"]
                })

            subset_format_data = {
                "GSM_FOBAR": format_data["GSM_FOBAR"],
                "MATH_Rephrased": format_data["MATH_Rephrased"],
                "GSM_Rephrased": format_data["GSM_Rephrased"]
            }
            json.dump(subset_format_data, open("raw_data/mmiqc-format-data-subset.json", "w"))

    def question_to_source():
        gsm8k_train_data = json.load(open("raw_data/gsm8k-format-data.json"))["train"]
        math_step_dpo_data = json.load(open("raw_data/math-step-dpo-format-data.json"))
        mmiqa_data = json.load(open("raw_data/mmiqc-format-data-subset.json"))

        question_to_source = defaultdict(list)
        for item in gsm8k_train_data:
            q = item["question"]
            s = "gsm8k_train"
            question_to_source[q].append(s)

        for item in math_step_dpo_data:
            q = item["question"]
            s = item["source"]
            question_to_source[q].append(s)

        for s in mmiqa_data:
            for item in mmiqa_data[s]:
                q = item["question"]
                question_to_source[q].append(s)

        print(len(question_to_source))
        json.dump(question_to_source, open("math_reasoning/question-to-source.json", "w"))

        # source = item["source"]
        # pattern = r'_(\d+)'
        ## Split the text using the regex pattern
        # match = re.search(pattern, source)

    # gsmplus()
    # gsm8k()
    # math_step_dpo()
    # mmiqa()
    # question_to_source()

    # ['gsm8k_train', 'MATH_Rephrased', 'gpt-4-1106-preview-MATH', 'gpt-3.5-turbo-GSM', 'GSM_Rephrased',
    # 'MATH_AnsAug', 'gpt-3.5-turbo-MATH', 'gpt-4-1106-preview-GSM', 'aqua', 'GSM_FOBAR']
    # all
    # gsm8k_train: 7473
    # MATH_Rephrased: 18759
    # GSM_FOBAR: 16503
    # aqua: 4851
    # =======
    # gpt-4-1106-preview-MATH: 99
    # gpt-3.5-turbo-GSM: 1565
    # GSM_Rephrased: 59275
    # MATH_AnsAug: 175
    # gpt-3.5-turbo-MATH: 30
    # gpt-4-1106-preview-GSM: 3

    # mistake = json.load(open("math_reasoning/"))

    prompt_hist = []
    question_hist = []

    if setting == "general":
        gsm8k_train = json.load(open("raw_data/gsm8k-format-data.json"))["train"]
        mmiqc_GSM_FOBAR = json.load(open("raw_data/mmiqc-format-data-subset.json"))["GSM_FOBAR"]
        mmiqc_GSM_Rephrased = json.load(open("raw_data/mmiqc-format-data-subset.json"))["GSM_Rephrased"]
        total_data = gsm8k_train + mmiqc_GSM_FOBAR + mmiqc_GSM_Rephrased
        print(len(total_data))
    else:
        gsm8k_train = json.load(open("math_reasoning/llama3-8b-instruct_initial_test/8shot_test_on_gsm8k_train.json"))
        mmiqc_GSM_FOBAR = json.load(open("math_reasoning/llama3-8b-instruct_initial_test/8shot_test_on_mmiqc_GSM_FOBAR.json"))
        mmiqc_GSM_Rephrased = json.load(open("math_reasoning/llama3-8b-instruct_initial_test/8shot_test_on_mmiqc_GSM_Rephrased-s0e40000.json"))
        total_data = gsm8k_train + mmiqc_GSM_FOBAR + mmiqc_GSM_Rephrased

    # ========================= general setting =========================
    # training setting
    warmup_train_data = []
    for idx, item in enumerate(tqdm(total_data)):
        if item["question"] in question_hist:
            continue

        if setting == "difficult" and item["match"] == "true":
            continue

        remain_items = total_data[:idx] + total_data[(idx + 1):]

        while True:
            combines = random.sample(remain_items, 3)
            combines.append(item)
            prompt, target = gsm8k_template(combines)

            if prompt not in prompt_hist:
                break

        item = {"prompt": prompt, "target": target}
        prompt_hist.append(prompt)
        question_hist.append(target)
        warmup_train_data.append(item)

    print("warmup training size: ", len(warmup_train_data))
    json.dump(warmup_train_data, open(f"math_reasoning/{setting}_data/warmup_train.json", "w"), indent=2)


    # preference setting
    iterative_dpo_data = []
    for _ in tqdm(range(500000)):
        while True:
            combines = random.sample(total_data, 4)
            prompt, target = gsm8k_template(combines)

            if prompt not in prompt_hist:
                break

        prompt_hist.append(prompt)
        item = {"prompt": prompt, "target": target}
        iterative_dpo_data.append(item)

    print("iterative training size: ", len(iterative_dpo_data))
    json.dump(iterative_dpo_data, open(f"math_reasoning/{setting}_data/iterative_preference_train.json", "w"), indent=2)


def format_custom_warmup_data(input_path,
                              warmup_train_size,
                              dpo_train_size=100000,
                              warmup_test_size=1000,
                              warmup_train_path="",
                              dpo_train_path="",
                              test_path=""
):
    '''
    :param input_path: path that saves the customized instructions
    :param warmup_train_size: the final training size for warming up
    :param hh_rlhf_dpo_train_size: the number of three-shot prompts to make proposer generate positive (make target model fails) or negative (cannot challenge the target model) data.
    :param warmup_test_size: the held-out test size for test the performance of proposer
    :return:
    '''

    instructions = json.load(open(input_path, "r"))  # todo: list of task instructions
    unique_prompts = []
    warmup_train_samples = []
    dpo_train_samples = []
    test_samples = []


    print(f"sample {warmup_train_size} training samples for warmup ...")

    if warmup_train_size > len(instructions):
        # make sure the outputs of each warmup sample are unique.
        warmup_train_size = len(instructions)

    for i in range(warmup_train_size):
        target = instructions[i]
        other_samples = instructions[:i] + instructions[(i+1):]
        demons = random.sample(other_samples, 3)
        demons.append(target)
        prompt, target = custom_template(demons)

        while prompt in unique_prompts:
            demons = random.sample(other_samples, 3)
            demons.append(target)
            # todo: you can design your own template
            prompt, target = custom_template(demons)

            while prompt in unique_prompts:
                prompt, target = custom_template(demons)

            unique_prompts.append(prompt)
            warmup_train_samples.append({
                "prompt": prompt,
                "target": target
            })

    print(f"sample {dpo_train_size} training samples for dpo ... and sample {warmup_test_size} samples for test ...")
    for size, store in zip([dpo_train_size, warmup_test_size], [dpo_train_samples, test_samples]):
        for i in range(size):
            demons = random.sample(instructions, 4)
            prompt, target = custom_template(demons)

            while prompt in unique_prompts:
                demons = random.sample(demons, 4)
                prompt, target = custom_template(demons)

                while prompt in unique_prompts:
                    demons = random.sample(demons, 4)
                    prompt, target = custom_template(demons)

                unique_prompts.append(prompt)
                store.append({
                    "prompt": prompt,
                    "target": target
                })

    json.dump(warmup_train_samples, open(warmup_train_path, "w"), indent=2)
    json.dump(dpo_train_samples, open(dpo_train_path, "w"), indent=2)
    json.dump(test_samples, open(test_path, "w"), indent=2)


if __name__ == '__main__':
    args.task = "math"

    if args.task == "safety":
        format_safe_warmup_data(
            hh_rlhf_warmup_train_size=8000,
            hh_rlhf_dpo_train_size=100000,
            hh_rlhf_warmup_test_size=800,
            toxic_warmup_train_size=1700,
            toxic_dpo_train_size=10000,
            toxic_warmup_test_size=300,
        )

        # deduplicate training data for warmup

    if args.task == "honesty":
        split_mmlu_by_question_category()  # subject --> samples

        format_honesty_warmup_data(
            dpo_size_per_cate=1700,
            test_size_per_cate=50
        )


    if args.task == "math":
        # format_math_warmup_data()
        format_math_warmup_data(setting="difficult")


    if args.task == "custom":
        format_custom_warmup_data(
            input_path="",
            warmup_train_size=10000,
            dpo_train_size=100000,
            warmup_test_size=1000,
            warmup_train_path="",
            dpo_train_path="",
            test_path=""
        )
