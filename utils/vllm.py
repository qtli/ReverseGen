import json
import pdb
import sys
import re
import os
import logging
from collections import defaultdict
import gc

MAX_INT = sys.maxsize
MAX_LENGTH = int(1024)  # Hardcoded max length to avoid infinite loop


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%d-%d %H:%M:%S')


def inference_vllm(config, processed_prompts, num_cpus=56, num_gpus=1):
    try:
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:
        pass

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info('>>>>>> one processed prompt:\n{}'.format(processed_prompts[0]))
    processed_prompts = processed_prompts[:config.sample_num] if config.sample_num > 0 else processed_prompts  # sample_num=-1
    logging.info('>>>>>> size of the processed prompts: {}\n'.format(len(processed_prompts)))
    import torch
    import ray
    # ray.shutdown()
    # https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
    logging.info('>>>>>> ray initialized')

    if "32" in config.torch_type:
        dtype = torch.float32
    else:
        # dtype = torch.float16
        dtype = "float16"


    llm = LLM(model=config.saved_policy,
              tensor_parallel_size=num_gpus,
              tokenizer=config.saved_policy,
              trust_remote_code=config.trust_remote_code,
              dtype=dtype,
              max_model_len=config.model.max_length)
    logging.info('>>>>>> model loaded')

    tokenizer = llm.get_tokenizer()

    # https://docs.vllm.ai/en/latest/offline_inference/sampling_params.html
    sampling_params = SamplingParams(temperature=config.temperature,
                                     # 0, Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.
                                     top_p=config.top_p,  # 1, Set to 1 to consider all tokens.
                                     max_tokens=config.model.max_new_length,
                                     # 2048, Maximum number of tokens to generate per output sequence.
                                     presence_penalty=config.presence_penalty,
                                     frequency_penalty=config.frequency_penalty,
                                     # logprobs=args.logprobs,
                                     # prompt_logprobs=args.logprobs,
                                     stop_token_ids=[tokenizer.eos_token_id,
                                                     tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                                     if config.model_name.lower() in ["llama3.1-8b-instruct", "llama3-8b-instruct"] else []  # KEYPOINT HERE
                                     )

    if config.num_try <= 1:
        outputs = llm.generate(processed_prompts, sampling_params)
        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
    else:
        pp_outputs = llm.generate(processed_prompts * int(config.num_try), sampling_params)
        pp_sorted_outputs = sorted(pp_outputs, key=lambda output: int(output.request_id))

        # grouping
        sorted_outputs=defaultdict(list)
        for out in pp_sorted_outputs:
            sorted_outputs[out.prompt].append(out)

    logging.info('>>>>>> generation done')

    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm  # Isn't necessary for releasing memory, but why not
    gc.collect()
    torch.cuda.empty_cache()
    return sorted_outputs
