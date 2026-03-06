import os, sys, torch, math, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, IO, BinaryIO, Tuple, Callable
from torch import Tensor, multinomial
from einops import reduce, einsum, rearrange
from jaxtyping import Float, Bool, Int

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str,str], dict[str,float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    save_path: str = None,
) -> None:
    '''
    Evaluate a LLM on a list of prompts and compute the evaluation metrics and then serialize the results to disk.
    '''
    results = [] # list[dict]
    outputs = vllm_model.generate(prompts, eval_sampling_params) # generate the outputs
    num_prompts = len(prompts)
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    for output, ground_truth in zip(outputs, ground_truths): # iterate over the outputs
        prompt = output.prompt
        generated_text = output.outputs[0].text

        # compute the reward
        reward = reward_fn(generated_text, ground_truth)

        results.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'ground_truth': ground_truth,
            'reward': reward,
        })

        if 'reward' in reward:
            total_reward += reward['reward']
        if 'format_reward' in reward:
            total_format_reward += reward['format_reward']
        if 'answer_reward' in reward:
            total_answer_reward += reward['answer_reward']
    
    avg_reward = total_reward / num_prompts
    avg_format_reward = total_format_reward / num_prompts
    avg_answer_reward = total_answer_reward / num_prompts
    print(f"Average reward: {avg_reward:.4f}, Average format reward: {avg_format_reward:.4f}, Average answer reward: {avg_answer_reward:.4f}")

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            for obj in results:
                f.write(json.dumps(obj) + '\n') # Write each object as a separate line
    return results

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Loading model...")
    model_path = 'Qwen/Qwen2.5-Math-1.5B'
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        n=1,
        min_tokens=4,
        max_tokens=1024, 
        temperature=1.0, 
        top_p=1.0, 
        stop=['</answer>'],
        include_stop_str_in_output=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading test data...")
    
    test_data_path = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/test.json"
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

    # load the R1 prompts
    r1_prompts_path = "/home/ruiqizhang/CS336/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"

    # Whether we use the chat template.
    use_chat_template = False
    if not use_chat_template:
        with open(r1_prompts_path, 'r') as f:
            r1_prompts = f.read()
        prompts = [r1_prompts.format(question=item['question']) for item in test_data]
        
    elif use_chat_template:
        with open(r1_prompts_path, 'r') as f:
            r1_prompts = f.readline() # only read the first line. Use the system prompt.
        prompts = []
        for item in test_data:
            messages = [
                {"role": "system", "content": r1_prompts},
                {"role": "user", "content": item["question"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

    ground_truths = [item['answer'] for item in test_data]

    # generate the results
    print("Evaluating the model...")
    results = evaluate_vllm(
        llm, 
        r1_zero_reward_fn, 
        prompts, 
        ground_truths, 
        sampling_params, 
        save_path="/home/ruiqizhang/CS336/assignment5-alignment/results/math_baseline_no_chat_template_Qwen2.5-Math-1.5B.json"
    )


    