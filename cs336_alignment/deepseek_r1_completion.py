import os, requests, json
from typing import List, Tuple, Dict, Any, Optional, IO, BinaryIO, Tuple, Callable, Iterable
from tqdm import tqdm

import openai
from openai import OpenAI
import asyncio

import re

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

os.environ['DEEPSEEK_API_KEY'] = 'sk-d157e7c624b644498861a1210bfd5a47'

INPUT_TOKENS_PER_MILLION = 0.2
CACHED_TOKENS_PER_MILLION = 2.0
OUTPUT_TOKENS_PER_MILLION = 3.0

def compute_cost(
    response: openai.types.chat.chat_completion.ChatCompletion,
    model_name: 'str',
) -> float:
    if model_name == 'deepseek':
        input_tokens = response.usage.prompt_tokens / 1e6
        cached_tokens = response.usage.prompt_tokens_details.cached_tokens / 1e6
        output_tokens = response.usage.completion_tokens / 1e6
        cost = INPUT_TOKENS_PER_MILLION * input_tokens + CACHED_TOKENS_PER_MILLION * cached_tokens + OUTPUT_TOKENS_PER_MILLION * output_tokens
        return round(cost, 8)
    else:
        raise ValueError(f'Invalid Model Name {model_name}.')

FORMAT_RE = re.compile(
    r'(</think>)\s*(<answer>)',
    re.IGNORECASE
)

# Correct the answer format to match the exact format check in the grader.
# Note: this is only used in querying the data, but not in the SFT, EI, or RL training.
def correct_answer_format(
    text: str
) -> str:
    return FORMAT_RE.sub('</think> <answer>', text)

def get_single_solution_deepseek(
    client: OpenAI,
    system_prompt: str,
    question: str,
    solution: str,
    reward_fn: Callable[[str,str], dict[str,float]],
    max_repeat_time: int = 1,
):
    generation_time = 0
    cost = 0
    correct = False
    while generation_time < max_repeat_time:
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{question}"},
                ],
                stream=False
            )
        except Exception as e:
            print(f'Fail to generte the solution to this questio. {e}.')
            continue

        generated_answer = response.choices[0].message.content
        generated_answer = correct_answer_format(generated_answer).strip() # correct the answer format as it will sometimes output </think>\n\n<answer> and we do not want this. And strip the leading and trailing whitespace.
        reward = reward_fn(generated_answer, solution)
        cost += compute_cost(response, 'deepseek')

        if reward['reward'] == 0.0:
            generation_time += 1
            continue
        else:
            record = {
                'question': question,
                'generated_answer': generated_answer,
                'solution': solution,
                'correct': True,
                'cost': cost
            }
            correct = True
            break
    
    if not correct: # we did not sample the correct answer.
        record = {
            'question': question,
            'generated_answer': generated_answer,
            'solution': solution,
            'correct': False,
            'cost': cost
        }

    return record

async def _run_one(sema, fn, kwargs: Dict[str, Any]):
    '''
    Run one sync call oin a thread with concurrency limit.
    sema is an asyncio.Semaphore. It limits how many async tasks can enter a critical section at the same time.
    '''
    async with sema: # RZ: Acquires a semaphore slot. If too many tasks are already running, this waits until a slot frees up. This is the concurrency limit.
        return await asyncio.to_thread(fn, **kwargs) # Runs the synchronous function fn in a background thread so it doesn’t block the event loop.
        # to_thread returns an awaitable that completes when the function finishes.

async def run_in_parallel(
    items: Iterable[Dict[str, Any]],
    max_workers: int,
    fn,
): 
    '''
    items: list of kwargs dicts for get_single_solution_deepseek
    max_workers: max concurrent requests
    fn: function to call (get_single_solution_deepseek)
    '''
    sema = asyncio.Semaphore(max_workers)
    tasks = [asyncio.create_task(_run_one(sema, fn, item)) for item in items]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await fut)
        # fut: it’s an asyncio.Task (a subclass of Future).
        # await fut: waits until that task finishes and returns its result (the return value of _run_one, which is the result of your get_single_solution_deepseek call). If the task raised, await re‑raises that exception.
        # asyncio.as_completed(tasks): returns an iterator of tasks in the order they finish (fastest first), not the order you submitted them.
        # The loop does not run them one by one — it just collects results as tasks finish. The actual concurrency is controlled by the semaphore in _run_one (and by the thread pool).
    return results

if __name__ == '__main__':
    # R1-zero prompt.
    r1_prompts_path = "/home/ruiqizhang/CS336/assignment5-alignment/cs336_alignment/prompts/deepseek_query.prompt"
    with open(r1_prompts_path, 'r') as f:
        r1_prompts = f.readline().strip()
        # RZ: Only use the first line of the prompt as the system prompt when we query from the DeepSeek API.
        # use .strip() to remove the trailing newline character.

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )

    test_data_path = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/train.json"
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    questions = [item['question'] for item in test_data]
    solutions = [item['answer'] for item in test_data]

    reward_fn = r1_zero_reward_fn

    items = [
        {
            'client': client,
            'system_prompt': r1_prompts,
            'question': question,
            'solution': solution,
            'reward_fn': reward_fn,
            'max_repeat_time': 3,
        }
        for question, solution in zip(questions, solutions)
    ]

    results = asyncio.run(run_in_parallel(items, max_workers=32, fn=get_single_solution_deepseek))
    
    save_path = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/sft_train.json"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        for obj in results:
            f.write(json.dumps(obj) + '\n') # Write each object as a separate line


# run it using nohup: nohup uv run python ./cs336_alignment/deepseek_r1_completion.py > ./logs/deepseek_r1_completion.log 2>&1 &
    