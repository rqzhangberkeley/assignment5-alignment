import os, sys, json, torch, math, random, time, re, logging, argparse, wandb
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, IO, BinaryIO, Tuple, Callable
from torch import Tensor, multinomial
from einops import reduce, einsum, rearrange
from jaxtyping import Float, Bool, Int

torch.set_float32_matmul_precision('high')

from cs336_alignment.sft_utils import tokenizer_prompt_and_output
from cs336_alignment.sft_utils import compute_entropy
from cs336_alignment.sft_utils import get_response_log_probs
from cs336_alignment.sft_utils import masked_normalize
from cs336_alignment.sft_utils import sft_microbatch_train_step
from cs336_alignment.sft_utils import init_vllm, load_policy_into_vllm_instance

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm

from vllm import LLM, SamplingParams

def setup_logger(logger_name: str):
    logger = logging.getLogger(logger_name) # a logging.Logger object. Name it.
    logger.setLevel(logging.INFO) # set the minimum severity level for this logger.
    logger.handlers.clear() # remove any existing handlers to avoid duplicate logs.

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    # return a logging.Formatter object which defines the logging format.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger

def build_parser():
    parser = argparse.ArgumentParser(
        description="SFT training script."
    )

    # device
    parser.add_argument('--policy_device', type = str, default = 'cuda:0', help = 'The device for the policy model.')
    parser.add_argument('--vllm_device', type = str, default = 'cuda:1', help = 'The device for the vllm model which is to do the validation.')
    parser.add_argument('--model_id', type = str, default = "Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument('--seed', type = int, default = 42)

    # data
    parser.add_argument('--train_data_path', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/sft_train.json")
    parser.add_argument('--val_data_path', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/sft_test.json")
    parser.add_argument('--max_seq_length_training', type = int, default = 1024)
    parser.add_argument('--max_seq_length_validation', type = int, default = 1024, help = "Maximum sequence length that we can generate.")
    parser.add_argument('--gpu_memory_utilization', type = float, default = 0.85)

    # training
    parser.add_argument('--global_batch_size', type = int, default = 16)
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 4)
    parser.add_argument('--num_epochs', type = int, default = 1)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--max_grad_norm', type = float, default = 1.0)

    # log
    parser.add_argument('--logging_steps', type = int, default = 1)
    parser.add_argument('--save_steps', type = int, default = 250)
    parser.add_argument('--ckpts_dir', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/ckpts")
    parser.add_argument('--run_name', type = str, default = 'sft_run_test')

    # eval
    parser.add_argument('--eval_steps', type = int, default = 50)
    parser.add_argument('--reward', type = str, default = 'drgrpo')
    parser.add_argument('--system_prompt_path', type = str, default = '/home/ruiqizhang/CS336/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt')
    parser.add_argument('--eval_save_dir', type = str, default = '/home/ruiqizhang/CS336/assignment5-alignment/results/sft')

    return parser

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def collate_fn(batch, system_prompt):
    # batch is a list of samples from your dataset.
    prompts = [system_prompt.format(question=x["question"]) for x in batch]
    outputs = [x["generated_answer"] for x in batch] # Use the CoT
    return prompts, outputs

def main(args):

    # init
    logger = setup_logger('SFT a small model on math dataset.......')
    wandb.init(project="cs336-sft", name=args.run_name)

    assert args.reward == 'drgrpo', f'Invalid reward function at validation: {args.reward}.'
    assert args.global_batch_size % args.gradient_accumulation_steps == 0, f'The global batch size must be a multiple of the gradient accumulation step. Get global batch size = {args.global_batch_size} and grad accum step = {args.gradient_accumulation_steps}.'
    actual_batch_size = args.global_batch_size // args.gradient_accumulation_steps
    grad_accum = args.gradient_accumulation_steps

    # system prompt: used in the validation and training. We should keep this fixed.
    with open(args.system_prompt_path, 'r') as f:
        system_prompt = f.read()

    # build the dataset.
    logger.info(f"Loading train data from {args.train_data_path}...")
    logger.info(f"Loading validation data from {args.val_data_path}...")
    train_data = load_jsonl(args.train_data_path)
    val_data = load_jsonl(args.val_data_path)
    train_loader = DataLoader(train_data, batch_size=args.global_batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, system_prompt))
    # DataLoader class will return a iterator. And it can receive either a list of dict or an obejct of torch.utils.data.Dataset class with __len__ and __getitem__ (if so, we shall define __getitem__ function to mke each item in self.data to be a dict or a tuple). The collator function can pack a sublist of dict in the data. 

    # total steps
    total_steps = math.ceil(len(train_data) / args.global_batch_size) * args.num_epochs

    # tokenizer: tokenizers ar CPU objects and do not have .to() method so we do not need to move it to GPUs.
    logger.info(f"Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # policy model
    logger.info(f"Loading policy model from {args.model_id}...")
    policy = AutoModelForCausalLM.from_pretrained(args.model_id).to(args.policy_device)
    
    # vllm instance
    logger.info(f"Initializing vllm instance from {args.model_id}...")
    vllm_model = init_vllm(model_id=args.model_id, device=args.vllm_device, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(
        max_tokens=args.max_seq_length_validation,
        temperature=1.0,
        top_p=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # reward function: used in the validation
    if args.reward == 'drgrpo':
        reward_fn = r1_zero_reward_fn

    # optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    logger.info(f"Starting training loop...")
    policy.train()
    global_step = 0 # the number of steps that we have completed
    accum_nll_sum = 0.0
    accum_entropy_sum = 0.0
    accum_token_count = 0.0

    start_time = time.time()
    for epoch in range(args.num_epochs):
        for prompts, outputs in train_loader:
            batch_size = len(prompts)
            effective_accum = math.ceil(batch_size / actual_batch_size) # RZ: For other battches except the last one, this should equals to grad_accum. This is only for the last batch where the batch size can be smaller than global_batch_size.

            for micro_start in range(0, batch_size, actual_batch_size):
                micro_prompts = prompts[micro_start:micro_start + actual_batch_size]
                micro_outputs = outputs[micro_start:micro_start + actual_batch_size]

                # tokenizer the prompts and the outputs
                batch = tokenizer_prompt_and_output(
                    prompt_strs=micro_prompts,
                    output_strs=micro_outputs,
                    tokenizer=tokenizer,
                    max_len=args.max_seq_length_training,
                )
                input_ids = batch['input_ids'].to(args.policy_device)
                labels = batch["labels"].to(args.policy_device)
                response_mask = batch["response_mask"].to(args.policy_device) # bool

                # get the log P and the entropy
                result = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True
                )
                log_probs = result['log_probs'] # B, T 
                token_entropy = result['token_entropy'] # B, T

                # loss and backward
                response_token_count = response_mask.float().sum().clamp_min(1)
                nll_sum = -1.0 * (log_probs * response_mask.float()).sum()
                loss = nll_sum / response_token_count / effective_accum
                loss.backward() # On each loss.backward(), PyTorch adds the new gradients into the existing p.grad buffer (i.e., p.grad += new_grad). It does not overwrite unless you clear it.

                accum_nll_sum += nll_sum.item()
                accum_entropy_sum += token_entropy[response_mask].sum().item()
                accum_token_count += response_token_count.item()
            
            is_last_step = (global_step+1) == total_steps
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm) # clip the gradient
            optimizer.step()
            optimizer.zero_grad()

            # logging.
            if (global_step+1)%args.logging_steps == 0 or is_last_step:
                avg_loss = accum_nll_sum / max(accum_token_count, 1.0)
                avg_entropy = accum_entropy_sum / max(accum_token_count, 1.0)

                current_time = time.time()
                time_per_step = (current_time - start_time) / (global_step + 1)
                estimated_time_to_completion = time_per_step * (total_steps - global_step - 1)
                logger.info(
                    f"step={global_step}, loss={avg_loss:.4f}, ent={avg_entropy:.4f}, time_per_step={time_per_step:.4f} seconds, estimated_time_to_completion={estimated_time_to_completion:.4f}."
                )
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/avg_entropy": avg_entropy,
                    }, 
                step=global_step   # this sets the x-axis
                )

                accum_nll_sum = 0.0
                accum_entropy_sum = 0.0
                accum_token_count = 0.0

            # validation
            if (global_step+1)%args.eval_steps == 0 or is_last_step:
                # sync policy -> vLLM
                load_policy_into_vllm_instance(policy, vllm_model)

                # validation data
                val_prompts = [system_prompt.format(question=item['question']) for item in val_data]
                val_solutions = [item['solution'] for item in val_data]
                save_path = os.path.join(
                    args.eval_save_dir,
                    f'lr_{args.lr}_B_{args.global_batch_size}_wd_{args.weight_decay}_step_{global_step+1}.json'
                )
                results = evaluate_vllm(
                    vllm_model, 
                    reward_fn, 
                    val_prompts, 
                    val_solutions, 
                    sampling_params, 
                    save_path
                )

                avg_reward = sum([result['reward']['reward'] for result in results]) / len(results)
                avg_format_reward = sum([result['reward']['format_reward'] for result in results]) / len(results)
                logger.info(
                    f"validation:step={global_step} avg_reward={avg_reward:.4f} avg_format_reward={avg_format_reward:.4f}."
                )
                wandb.log(
                    {
                        "val/avg_reward": avg_reward,
                        "val/avg_format_reward": avg_format_reward,
                    }, 
                step=global_step   # this sets the x-axis
                )

            # save the ckpts.
            if (global_step+1)%args.save_steps == 0 or is_last_step:
                os.makedirs(args.ckpts_dir, exist_ok=True)
                torch.save(policy.state_dict(), os.path.join(args.ckpts_dir, f'sft_lr_{args.lr}_B_{args.global_batch_size}_wd_{args.weight_decay}_step_{global_step+1}.pth'))
                logger.info(f"Saved checkpoint at step {global_step+1}.")
            
            global_step += 1

if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)

# Run this script using python instead of torchrun since the script is single‑process and just uses two GPUs for different roles. torchrun is only needed for DDP/multi‑process training.
# nohup uv run python ./cs336_alignment/sft.py --run_name sft_baseline_lr_1e-5_B_16_wd_0.0 > ./logs/sft.log 2>&1 &
# nohup uv run python ./cs336_alignment/sft.py --run_name sft_baseline_lr_1e-5_B_16_wd_0.01 --policy_device cuda:2 --vllm_device cuda:3 > ./logs/sft_wd_0.01.log 2>&1 &