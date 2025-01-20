# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import transformers
from transformers import set_seed
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import json
from scipy.stats import entropy
from datetime import datetime
from utils.common_prompt import wrap_instructions

def select_data(config, data):
    if config.start_idx != -1 or config.end_idx != -1:
        data = data[config.start_idx:config.end_idx]

    return data

def load_data(config, data):
    prompts, instructions, ref_solutions = [], [], []

    if "gsm8k" in config.test_file:
        data = json.load(open(config.test_file, "r"))
        data = data["test"]
        for item in data:
            instructions.append(item["question"])
            ref_solutions.append(item["answer"])
        prompts = [""] * len(ref_solutions)

    elif "gsmplus" in config.test_file:
        data = json.load(open(config.test_file, "r"))
        for item in data:
            instructions.append(item["question"])
            ref_solutions.append(item["solution"])
        prompts = [""] * len(instructions)

    elif "evol_inst" in config.test_file:
        data = json.load(open(config.test_file, "r"))
        for item in data:
            instructions.append(item[config.prompt_key])
            ref_solutions.append(item[config.target_key])

        prompts = [""] * len(instructions)

    elif "proposer" in config.test_file:
        data = json.load(open(config.test_file, "r"))
        for item in data:
            instructions.append(item[config.prompt_key])
            # ref_solutions.append(item["predict_solution"])
        prompts = [""] * len(ref_solutions)
        ref_solutions = [""] * len(instructions)

    else:
        for line in data.readlines():
            item = json.loads(line)
            instructions.extend(item["policy"])
            prompts.extend(item['prompt'] * len(item["policy"]))
            ref_solutions.extend([""] * len(item["policy"]))


    if config.start_idx != -1 or config.end_idx != -1:
        prompts = select_data(config, prompts)
        instructions = select_data(config, instructions)
        ref_solutions = select_data(config, ref_solutions)

    return prompts, instructions, ref_solutions


def generate_response(config: DictConfig):
    from utils.vllm import inference_vllm
    # We use vllm to generate response since all current target models are supported by vllm.
    data = open(config.test_file, "r")

    if config.start_idx != -1 or config.end_idx != -1:
        opt_file = ".".join(config.output_file.split(".")[:-1]) + f"_{config.start_idx}_{config.end_idx}.json"
    else:
        opt_file = config.output_file
    if os.path.exists(opt_file):
        print(f"{opt_file} already exists!!")
        return [], opt_file
    print(f"writing to {opt_file}...")

    prompts, instructions, ref_solutions = load_data(config, data)
    # prompts: prompt for generating instructions
    # instructions: fed into the target model
    # ref_solutions: golden answers for the instructions

    # wrap instructions into suitable task prompts to target model
    templated_instructions, clean_instructions, clean_solutions = wrap_instructions(config, prompts, instructions, ref_solutions)

    print(f"The total number of instructions ", len(templated_instructions))
    print("=" * 50)
    print(templated_instructions[0])
    print("=" * 50)
    print(templated_instructions[1])

    predictions = inference_vllm(config,
                                 processed_prompts=templated_instructions,
                                 num_cpus=56,
                                 num_gpus=1)  # todo: You can adjust 1 to reflect your actual available GPU count.
    print('size of predictions: ', len(predictions))

    outputs = []
    for idx, output in enumerate(predictions):
        if config.num_try == -1:
            model_prediction = output.outputs[0].text
        else:
            model_prediction = [o.outputs[0].text for o in predictions[output]]

        item_format = {
            'idx': idx,
            'predict_instruction': clean_instructions[idx],
            # 'prompt': output.prompt if config.num_try<=1 else output,
            "target_model_response": model_prediction
        }

        if "honesty" in config.test_file:
            occurance = {}
            for ans in model_prediction:
                if ans in occurance:
                    occurance[ans] += 1
                else:
                    occurance[ans] = 1
            freq_list = list(occurance.values())
            answer_entropy = entropy(freq_list)
            item_format["response_entropy"] = answer_entropy

        if "math" in config.test_file or "gsm" in config.test_file or "init_train" in config.test_file:
            # For math reasoning, we need golden answers to calculate accuracy.
            item_format["gold_response"] = clean_solutions[idx]

        outputs.append(item_format)

    print(outputs[0])
    json.dump(outputs, open(opt_file, 'w'), indent=2)
    return outputs, opt_file


@hydra.main(version_base=None, config_path="config", config_name="config")
def main_infer(config: DictConfig):
    """Main entry point for evaluating. Validates config, loads model(s), and kicks off worker process(es)."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
    print(OmegaConf.to_yaml(config))

    if config.use_gpus != "all":
        print("use GPU: ", config.use_gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.use_gpus)

    if config.mode not in ['sample_inst', 'respond']:
        raise Exception("This is a script for sampling instructions or obtaining feedback. config.mode should be one of 'sample_inst', or 'respond'.")

    if config.mode == 'respond':
        # generate responses of target model based on the sampled instructions of proposer model
        generate_response(config)
        return

    set_seed(config.seed)
    print('=' * 80)
    print(f'Writing to', config.samples_dir)
    print('=' * 80)
    # purely inference, so put as much as possible onto the first gpu
    # an alternative to auto, distributes data around all GPUs but the first one,
    # putting only the remainder on the first GPU and reserving most or all of the first GPU's memory for inference
    model_kwargs = {'device_map': "balanced_low_0"}


    if config.start_idx != -1 or config.end_idx != -1:
        file_suffix = f"_s{config.start_idx}e{config.end_idx}"
    else:
        file_suffix = ""

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    from utils.common_funcs import disable_dropout, rank0_print
    from trainers import BasicTrainer, DPOTrainer
    import dataloader

    # first see if saved tokenizer is in the experiment directory
    tokenizer_name_or_path = config.local_run_dir or config.model.tokenizer_name_or_path
    print(f'Loading tokenizer at {tokenizer_name_or_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        # since the human/asst fields are not in the configs of the already-released models, add defaults
        human_prefix=config['human_prefix'],
        human_suffix=config['human_suffix'],
        assistant_prefix=config['assistant_prefix'],
        assistant_suffix=config['assistant_suffix'],
        seed=config.seed,
        # the following kwargs can be used to make explore_dataset imbalanced (only used by UnbalancedUnpairedPreferenceDataLoader)
        frac_unique_desirable=config.get('frac_unique_desirable', 1.0),
        frac_unique_undesirable=config.get('frac_unique_undesirable', 1.0),
        # control tokens taken from Korbak et al.'s (2023) "Pretraining Models with Human Feedback"
        chosen_control_token=(config.loss.chosen_control_token if config.loss.name == "csft" else None),
        rejected_control_token=(config.loss.rejected_control_token if config.loss.name == "csft" else None),
    )

    os.makedirs(config.samples_dir, exist_ok=True)
    os.makedirs(os.path.join(config.samples_dir, config.exp_name), exist_ok=True)
    f_config = f'{config.exp_name}_top_{config.top_p}_tmp_{config.temperature}_nt_{config.num_try}_{config.sample_file_name}{file_suffix}'
    fn = os.path.join(config.samples_dir, config.exp_name, f_config+".json")
    print(f'saving samples to {fn} ...')
    total_data = []

    # use the SFT dataloader because we don't want to repeat prompts
    # and bc data ordering is different in paired vs unpaired data loaders
    # this way, sampled prompts are the same for a given seed
    eval_iterator = dataloader.SFTDataLoader(
        config.datasets,
        tokenizer,
        split="test",
        batch_size=config.model.eval_batch_size,
        n_examples=config.n_samples,
        max_prompt_count=1,
        start_idx=config.start_idx,
        end_idx=config.end_idx,
        exp_name=config.exp_name,
        prev_fw=fn,
        config=config,
        **data_iterator_kwargs
    )

    if config.use_vllm:
        from utils.vllm import inference_vllm
        all_prompts, all_chosen, prompt_to_chosen = [], [], {}

        # num_gpus = torch.cuda.device_count()
        num_gpus = 1

        eval_batches = list(eval_iterator)
        for eval_batch in eval_batches:
            chosen_samples = []
            for x in (eval_batch['target_text'] if 'target_text' in eval_batch else eval_batch['chosen_text']):
                if tokenizer.eos_token in x:
                    x = x[:x.rfind(tokenizer.eos_token)]
                chosen_samples.append(x)
            all_prompts.extend(eval_batch['prompt_text'])
            all_chosen.extend(chosen_samples)

        for p, c in zip(all_prompts, all_chosen):
            prompt_to_chosen[p] = c

        if len(all_prompts) >= 1:
            predictions = inference_vllm(config, processed_prompts=all_prompts, num_cpus=56, num_gpus=num_gpus)
            for idx, output in enumerate(predictions):
                if config.num_try <= 1:
                    model_prediction = [output.outputs[0].text]
                    prompt = output.prompt
                else:
                    prompt = output
                    model_prediction = [o.outputs[0].text for o in predictions[prompt]]

                for mp in model_prediction:
                    item_format = {
                        'idx': idx,
                        'prompt': prompt,
                        'chosen': prompt_to_chosen[prompt],
                        config.target_key: mp
                    }
                    total_data.append(item_format)
            json.dump(total_data, open(fn, "w"), indent=2)

    else:
        # using model.generate() for inference
        print('building policy')
        policy_dtype = getattr(torch, config.model.policy_dtype)
        use_safetensors = False
        for name in os.listdir(config.model.name_or_path):
            if name.endswith('.safetensors'):
                use_safetensors = True
                break
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, use_safetensors=use_safetensors, low_cpu_mem_usage=True,
            use_flash_attention_2=config.model.use_flash_attention, torch_dtype=policy_dtype, **model_kwargs)
        # note that models were only resized for csft before saving
        # important because number of tokens in pretrained tokenizer is different from model.config.vocab_size,
        # so resizing at eval will throw an error if not resized before training
        if config.loss.name == 'csft':
            policy.resize_token_embeddings(len(tokenizer))  # model being loaded should already be trained with additional tokens for this to be valid
        disable_dropout(policy)

        # saved policy can be force set to null to sample from pretrained model
        if config.saved_policy is not None:
            state_dict = torch.load(config.saved_policy, map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(
                f'loading pre-trained weights for policy at step {step} from {config.saved_policy} with metrics {json.dumps(metrics, indent=2)}')
            policy.load_state_dict(state_dict['state'])

        config.n_samples = eval_iterator.n_examples
        trainer = BasicTrainer(tokenizer, config, None, eval_iterator, policy, reference_model=None)

        if "honesty" in config.local_run_dir:
            # First, the model's chosen option is collected, then the associated "sure" probability is recorded.
            gen_uncertainty = True if "certain" in config.sample_file_name else False
            trainer.sample(gen_mmlu_train=True, gen_uncertainty=gen_uncertainty, fn=fn)
        else:
            trainer.sample(fn=fn)

    date_format = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    config_fn = os.path.join(config.samples_dir, config.exp_name, f'{f_config}_{date_format}_with_config.json')

    json.dump(
        {
        'sampled_at': str(datetime.now()),
        'config': OmegaConf.to_container(config, resolve=True)
            },
        open(config_fn, 'w'), indent=2)
    print(f"saving eval configs to {config_fn} ...")




if __name__ == '__main__':
    main_infer()
