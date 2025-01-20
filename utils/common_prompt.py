import json
import pdb
import sys
import re
import os
import random
from transformers import AutoTokenizer

project_path = os.getcwd()  # The path of the currently invoked outermost script
sys.path.append(project_path)
from utils.fastchat_conversation_template import get_conv_template



MODEL_NAME_TO_TEMPLATE_NAME = {
    "vicuna-7b-v1.5": "vicuna_v1.1",
    "llama-2-7b": "llama-2",
    "llama-2-7b-chat": "llama-2"
}


def wrap_query_to_prompt(conv, query):
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def sample_demonstrations(mi, to, each_sample_size, sample_num):
    sample_demons = []
    for _ in range(sample_num):
        mi_samples = random.sample(mi, int(each_sample_size))
        to_samples = random.sample(to, int(each_sample_size))
        sample_demons.append(mi_samples+to_samples)
    return sample_demons



def clean_query(p, q, data_source="hh"):
    global bad_cases_count
    if "safety" in data_source:
        pattern = r"\s([5-9]|[1-9]\d+)\."
        match = re.search(pattern, q)
        if match:
            split_parts = re.split(pattern, q)
            q = split_parts[0].strip()
        else:
            return q.strip()
    elif "honesty" in data_source:
        pattern = r"(\nAnswer:[A-D])"
        match = re.search(pattern, q)
        INST = "Ask multiple choice questions (with answers) about"

        if match:

            if len(p.split(INST)) >= 2 and len(p.split(INST)[1].split(":\n\n")) >= 1:
                split_parts = re.split(pattern, q)[:2]
                q = "".join(split_parts).strip()+"."
            else:
                q = ""
        else:
            q = ""
            if len(p.split(INST)) >= 2 and len(p.split(INST)[1].split(":\n\n")) >= 1:
                    subject = p.split(INST)[1].split(":\n\n")[0].strip()
                    # print("prompt: ", p, "\nquestion: ", q, "\n\n")
                    # bad_cases.write("prompt: " + p + "\nquestion: " + q + "\n===================================================\n\n")
                    bad_cases_count += 1
    else:
        pass
    return q



def next_option_letter(letter):
    if letter == 'Z':
        return 'A'
    else:
        return chr(ord(letter) + 1)


def get_mmlu_prompt_data():
    dir = os.path.join(os.getcwd().split("ReverseGen")[0], "ReverseGen")
    mmlu_prompt_data = json.load(open(os.path.join(dir, "explore_dataset", "raw_data", "MMLU", "MMLU_ID_prompt.json")))
    return mmlu_prompt_data


mmlu_prompt_data = get_mmlu_prompt_data()

def gen_mmlu_prompt(input_mcqa, input_prompt):
    choices = ["A", "B", "C", "D"]
    subject = input_prompt.split(" about")[1].split(":\n\n")[0].strip()
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_shots(prompt_data):
        prompt = ""
        for data in prompt_data:
            prompt += data[0]
            k = len(data) - 2
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], data[j + 1])
            if "none of" not in prompt.lower():
                pattern = re.compile(r"[A-F]\. ")
                last_option_letter = pattern.findall(prompt)[-1].replace(".", "").strip()
                nl = next_option_letter(last_option_letter)
                prompt += f"\n{nl}. None of the above"  # todo: 防止生成的题目和选项有问题
            prompt += "\nAnswer:"
            prompt += data[k + 1] + "\n\n"

        return prompt

    def format_example(input_list):
        prompt = input_list.split("\nAnswer:")[0].strip()
        if "none of" not in prompt.lower():
            pattern = re.compile(r"[A-F]\. ")
            if len(pattern.findall(prompt)) == 0:
                # 没有 ABCD四个选项
                return ""
            last_option_letter = pattern.findall(prompt)[-1].replace(".", "").strip()
            nl = next_option_letter(last_option_letter)
            # Add a "NONE" option for cases with all invalid choices.
            prompt += f"\n{nl}. None of the above\nAnswer:"
        return prompt

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject).strip())

    prompt += format_shots(mmlu_prompt_data["_".join(subject.split(" "))])
    fe = format_example(input_mcqa)
    if fe == "":
        return ""
    prompt += fe
    return prompt


def evaluate_mmlu(subject, question):
    question = question.split("\nAnswer:")[0].strip()
    if "none of" not in question.lower():
        pattern = re.compile(r"[A-D]\. ")
        if len(pattern.findall(question)) == 0:
            return ""
        last_option_letter = pattern.findall(question)[-1].replace(".", "").strip()
        nl = next_option_letter(last_option_letter)
        question += f"\n{nl}. None of the above"

    prompt = f"""Evaluate the validity of the given multiple choice question related to "{subject}" and provide a conclusion as "valid" or "invalid". If the question is deemed "valid", provide the correct answer in the format "Valid. Answer: [Option, e.g. A]", without including any further explanation or details.
Question: {question}
Evaluation: """
    return prompt


def evaluate_gsm8k(question):
    prompt = f"""Evaluate the validity of the given grade school math question and provide a conclusion as "valid" or "invalid". If the question is deemed "valid", provide the correct answer in the format "Valid. Solution: [Reasoning Process] #### [Answer Value]". Do not include any further explanation or details. If the question is deemed "invalid", begin your response with "Invalid".
Question: {question}
Evaluation: """
    return prompt

def evaluate_math(question):
    prompt = f"""Evaluate the validity of the given math question and provide a conclusion as "valid" or "invalid". If the question is deemed "valid", provide the correct answer in the format "Valid. Solution: [Reasoning Process] #### [Answer Value]". Do not include any further explanation or details. If the question is deemed "invalid", begin your response with "Invalid".
Question: {question}
Evaluation: """
    return prompt


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


def cot_prompt(question):

    gsm8k_nshot_prompt = ""
    for q, a in gsm8k_nshots:
        gsm8k_nshot_prompt += "Question:\n{input}\nAnswer:\nLet's think step by step.\n{answer}\n".format(input=q,
                                                                                                          answer=a)
    gsm8k_nshot_prompt += f"Question:\n{question}\nAnswer:\nLet's think step by step.\n"
    return gsm8k_nshot_prompt



def gsm_nshot_chats(n: int, question: str, config) -> dict:
    # https://medium.com/@sewoong.lee/how-to-reproduce-llama-3s-performance-on-gsm-8k-e0dce7fe9926
    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f"Answer:\nLet's think step by step.\n{s}"

    SYS_MES = "Your task is to solve a series of math word problems by providing the final answer. Use the format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560. Make sure to carefully read and understand each problem before providing your answer."
    mn_lower = config.model.name_or_path.lower()
    if "qwen" in mn_lower:
        SYS_MES = "Your task is to solve a series of math word problems by providing the final answer. Put your final answer within \\boxed{{}}. For example, if the answer is 560, you should write \\boxed{560}"

    chats = [
        # {"role": "system",
        #  "content": "Your task is to solve a series of math word problems by providing the final answer. Put your final answer within \\boxed{{}}. For example, if the answer is 560, you should write \\boxed{{560}}"}
        {"role": "system", "content": SYS_MES}
    ]

    # random.seed(42)
    # for qna in random.sample(nshot_data, n):
    for q, a in gsm8k_nshots[:n]:
        chats.append(
            {"role": "user", "content": question_prompt(q)})
        chats.append(
            {"role": "assistant", "content": answer_prompt(a)})

    # chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})
    chats.append({"role": "user", "content": question_prompt(question)})

    return chats


def wrap_instructions(config, prompts, instructions, solutions):
    mn_lower = config.name_or_path.lower()

    if "vicuna" in mn_lower or "llama-2" in mn_lower:
        # apply conversational template of specific models
        # "vicuna-7b-v1.5"
        # "llama-2-7b"
        # "llama-2-7b-chat"
        mn = "vicuna_v1.1" if "vicuna" in mn_lower else mn = "llama-2"
        if "math" in config.test_file or "gsm" in config.test_file:
            conv = get_conv_template(mn + "_math")
        else:
            conv = get_conv_template(mn)

    wrap_queries = []
    clean_queries = []
    clean_solutions = []
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    for idx, (p,q,s) in enumerate(zip(prompts, instructions, solutions)):
        q = clean_query(p, q, data_source=config.test_file)
        if q == "":
            continue

        if "-instruct" in mn_lower:
            # generate responses
            if config.nshots is not None:
                n = config.nshots
            else:
                n = 8
            if ("math" in config.test_file or "gsm" in config.test_file):
                prompt_template = gsm_nshot_chats(n=n, question=q, config=config)
            else:
                prompt_template = [
                    {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."},
                    {"role": "user", "content": q},
                ]
            wq = tokenizer.apply_chat_template(
                conversation=prompt_template,
                tokenize=False,
                add_generation_prompt=True,
            )

        elif "safety" in config.test_file:
            wq = wrap_query_to_prompt(conv, q)
            conv.messages = []

        elif "honesty" in config.test_file:
            wq = gen_mmlu_prompt(q, p)

        elif "evol_inst" in config.test_file:
            wq = q

        if wq == "":
            continue

        wrap_queries.append(wq)
        clean_queries.append(q)
        clean_solutions.append(s)

    return wrap_queries, clean_queries, clean_solutions


