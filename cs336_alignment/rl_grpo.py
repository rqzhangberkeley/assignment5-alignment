import os, sys, json, torch, math, random, time, re, logging, argparse, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from typing import Literal

torch.set_float32_matmul_precision('high')

from cs336_alignment.sft_utils import tokenizer_prompt_and_output
from cs336_alignment.sft_utils import get_response_log_probs
from cs336_alignment.sft_utils import init_vllm, load_policy_into_vllm_instance

from cs336_alignment.rl_utils import compute_group_normalized_rewards
from cs336_alignment.rl_utils import grpo_microbatch_train_step
from cs336_alignment.rl_utils import collate_fn_grpo
from cs336_alignment.rl_utils import masked_mean

# reward and evaluation functions
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm

from vllm import LLM, SamplingParams
os.environ['VLLM_DISABLE_TQDM'] = '1'

# helper functions from the SFT part.
from cs336_alignment.sft import setup_logger, load_jsonl

def build_parser():
    parser = argparse.ArgumentParser(
        description="Training a model using GRPO. This is the main training script."
    )

    # device
    parser.add_argument('--policy_device', type = str, default = 'cuda:1', help = 'The device for the policy model.')
    parser.add_argument('--vllm_device', type = str, default = 'cuda:2', help = 'The device for the vllm model which is to do the validation.')
    parser.add_argument('--model_id', type = str, default = "Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument('--seed', type = int, default = 42)

    # data. We use SFT data by default for simplicity, and we only use the prompts.
    parser.add_argument('--train_data_path', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/sft_train.json")
    parser.add_argument('--val_data_path', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/data/math/sft_test.json")
    parser.add_argument('--system_prompt_path', type = str, default = '/home/ruiqizhang/CS336/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt')
    parser.add_argument('--eval_save_dir', type = str, default = '/home/ruiqizhang/CS336/assignment5-alignment/results/grpo')
    parser.add_argument('--ckpts_dir', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/ckpts")

    # Inference.
    parser.add_argument('--sampling_min_tokens', type = int, default = 4, help = "Min sequence length that we can generate at training and validation time. This is to avoid the empty string.")
    parser.add_argument('--sampling_max_tokens', type = int, default = 1024, help = "Max sequence length that we can generate at training and validation time")
    parser.add_argument('--sampling_temperature', type = float, default = 1.0)
    parser.add_argument('--gpu_memory_utilization', type = float, default = 0.85)

    # Loss type
    parser.add_argument('--loss_type', type=str, choices=['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],default='reinforce_with_baseline')
    parser.add_argument('--use_std_normalization', action='store_true')
    parser.add_argument('--no_use_std_normalization', dest='use_std_normalization', action='store_false')
    parser.set_defaults(use_std_normalization=True)
    parser.add_argument('--advantage_eps', type = float, default = 1e-6)
    parser.add_argument('--cliprange', type = float, default = 0.2)

    # training
    parser.add_argument('--num_epochs', type = int, default = 1)
    parser.add_argument('--max_grpo_steps', type = int, default = 200, help = 'One step = one generation step. And it can contain multiple training steps if the training batch size < rollout batch size.')
    parser.add_argument('--rollout_batch_size', type = int, default = 256, help = 'The number of responses we generate for each GRPO step.')
    parser.add_argument('--train_batch_size', type = int, default = 256, help = 'The number of responses we train for each GRPO training step.')
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 128)
    parser.add_argument('--group_size', type = int, default = 8)

    # optimizer
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--max_grad_norm', type = float, default = 1.0)

    # log
    parser.add_argument('--logging_steps', type = int, default = 1)
    parser.add_argument('--save_steps', type = int, default = 250)
    parser.add_argument('--run_name', type = str, default = 'grpo_run_test')

    # eval
    parser.add_argument('--eval_steps', type = int, default = 50)
    parser.add_argument('--reward', type = str, default = 'drgrpo')

    return parser

def main(args):

    # init
    logger = setup_logger('Train a small model on math dataset using RL with Verifieable Rewards.......')
    wandb.init(project="cs336-grpo", name=args.run_name)

    # assert
    assert args.reward == 'drgrpo', f'Invalid reward function at validation: {args.reward}.'
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        'train_batch_size must be divisible by gradient_accumulation_steps'
    )
    n_micro_batches_per_train_batch = args.train_batch_size // args.gradient_accumulation_steps # the number of responses that we us to train at each training microbatch step.
    assert args.rollout_batch_size % args.group_size == 0, (
        'rollout_batch_size must be divisible by group_size'
    )
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size, (
        'train_batch_size must be larger or equal to the group size'
    )
    assert args.train_batch_size <= args.rollout_batch_size, (
        'train_batch_size must be <= rollout_batch_size'
    )
    assert args.rollout_batch_size % args.train_batch_size == 0, (
        'rollout_batch_size must be divisible by train_batch_size'
    )
    n_train_steps_per_rollout_batch = args.rollout_batch_size // args.train_batch_size

    # system prompt: used in the validation and training. We should keep this fixed.
    with open(args.system_prompt_path, 'r') as f:
        system_prompt = f.read()

    # tokenizer: tokenizers are CPU objects and do not have .to() method so we do not need to move it to GPUs.
    logger.info(f"Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # build the dataset.
    logger.info(f"Loading train data from {args.train_data_path}...")
    logger.info(f"Loading validation data from {args.val_data_path}...")
    train_data = load_jsonl(args.train_data_path)
    val_data = load_jsonl(args.val_data_path)
    train_loader = DataLoader(
        train_data,
        batch_size=n_prompts_per_rollout_batch,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_grpo(batch, system_prompt, tokenizer),
    )
    # RZ Notes: Here batch size is the number of prompts that we use for each rollout batch. 
    # Add system prompt in the data collator. The data collator will provide the solutions.

    # total steps
    steps_in_one_epoch = len(train_data) // n_prompts_per_rollout_batch
    total_steps = min(steps_in_one_epoch * args.num_epochs, args.max_grpo_steps)
    
    # policy model
    logger.info(f"Loading policy model from {args.model_id}...")
    policy = AutoModelForCausalLM.from_pretrained(args.model_id).to(args.policy_device)
    
    # vllm instance
    logger.info(f"Initializing vllm instance from {args.model_id}...")
    vllm_model = init_vllm(model_id=args.model_id, device=args.vllm_device, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)
    train_sampling_params = SamplingParams(
        n=args.group_size,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        temperature=args.sampling_temperature,
        top_p=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    val_sampling_params = SamplingParams(
        n=1,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        temperature=args.sampling_temperature,
        top_p=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
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
    is_off_policy_training = args.rollout_batch_size > args.train_batch_size
    start_time = time.time()
    # accumulate metrics across logging_steps
    meta_data = {"grad_norm": [], "loss": [], "entropy": []}

    for epoch in range(args.num_epochs):
        for prompts, solutions in train_loader:

            # generate responses. Currently, we do not get the log-P directly from the vllm.
            load_policy_into_vllm_instance(policy, vllm_model)
            outputs = vllm_model.generate(prompts, train_sampling_params)
            outputs_text = [output.outputs[i].text for output in outputs for i in range(args.group_size)] # pay attention to the order
            repeated_prompts = [p for p in prompts for _ in range(args.group_size)]
            repeated_solutions = [s for s in solutions for _ in range(args.group_size)]

            # compute the group normalized rewards
            advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=outputs_text,
                repeated_ground_truths=repeated_solutions,
                group_size=args.group_size,
                advantage_eps=args.advantage_eps,
                normalized_by_std=args.use_std_normalization
            )
            advantages = advantages.to(args.policy_device)
            raw_rewards = raw_rewards.to(args.policy_device)

            # tokenize
            rollout_batch = tokenizer_prompt_and_output(prompt_strs=repeated_prompts, output_strs=outputs_text, tokenizer=tokenizer,max_len=args.sampling_max_tokens)
            input_ids = rollout_batch['input_ids'].to(args.policy_device)
            labels = rollout_batch["labels"].to(args.policy_device)
            response_mask = rollout_batch["response_mask"].to(args.policy_device) # bool

            # If this is off-policy training, we need to store the old log probs before the training steps.
            if is_off_policy_training:
                old_log_probs = get_response_log_probs(model=policy, input_ids=input_ids, labels=labels, return_token_entropy=False)['log_probs'] # B, T
            else:
                old_log_probs = None

            # log the metrics in the metadata (accumulated across logging_steps)

            # Each train step
            for train_batch in range(n_train_steps_per_rollout_batch):

                # Each micro step.
                for micro_batch in range(args.gradient_accumulation_steps):
                    micro_batch_start = train_batch * args.train_batch_size + micro_batch * n_micro_batches_per_train_batch
                    micro_batch_end = micro_batch_start + n_micro_batches_per_train_batch
                    input_ids_micro_batch = input_ids[micro_batch_start:micro_batch_end]
                    labels_micro_batch = labels[micro_batch_start:micro_batch_end]
                    response_mask_micro_batch = response_mask[micro_batch_start:micro_batch_end]
                    raw_rewards_micro_batch = raw_rewards[micro_batch_start:micro_batch_end] # (B, )
                    advantages_micro_batch = advantages[micro_batch_start:micro_batch_end] # (B, )
                    if old_log_probs is not None: # If we are in the off-policy case.
                        old_log_probs_micro_batch = old_log_probs[micro_batch_start:micro_batch_end]
                    else:
                        old_log_probs_micro_batch = None

                    # get the log P and token entropy
                    log_probs_result = get_response_log_probs(
                        model=policy,
                        input_ids=input_ids_micro_batch,
                        labels=labels_micro_batch,
                        return_token_entropy=True,
                    )
                    log_probs = log_probs_result['log_probs']  # B, T
                    token_entropy = log_probs_result['token_entropy']  # B, T

                    # PG loss. The loss is already appliedwith loss.backward() in the function.
                    # Already apply /= accumulation_steps in the function.
                    loss, meta_data_micro_batch = grpo_microbatch_train_step(
                        policy_log_probs=log_probs,
                        response_mask=response_mask_micro_batch,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        loss_type=args.loss_type,
                        raw_rewards=raw_rewards_micro_batch.unsqueeze(-1).expand(-1, log_probs.shape[1]), # need (B, T). Same advantage for every token in the response.
                        advantages=advantages_micro_batch.unsqueeze(-1).expand(-1, log_probs.shape[1]),
                        old_log_probs=old_log_probs_micro_batch,
                        cliprange=args.cliprange
                    )

                    # log the metrics in the metadata
                    # All the metrics have already been averaged by /= gradient_accumulation_steps in the function.
                    meta_data['loss'].append(loss.item())
                    meta_data['entropy'].append(masked_mean(token_entropy, response_mask_micro_batch).item())
                    for key, value in meta_data_micro_batch.items():
                        if key not in meta_data:
                            meta_data[key] = []
                        meta_data[key].append(value)

                # clip the gradient; returns total norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.max_grad_norm,
                )
                meta_data["grad_norm"].append(grad_norm.item()) # One number every train step.
                optimizer.step()
                optimizer.zero_grad()

            # logging.
            if (global_step+1)%args.logging_steps == 0:
                current_time = time.time()
                time_per_step = (current_time - start_time) / (global_step + 1)
                estimated_time_to_completion = time_per_step * (total_steps - global_step - 1)

                # The loss is averaged over all non-masked token in one micro batch, and then averaged over all micro batch in one train step (sum(...) / gradient_accumulation_steps), and then averaged over all train steps in one rollout batch (sum(...) / n_train_steps_per_rollout_batch), and then averaged over all rollout batches in one logging step (sum(...) / args.logging_steps).
                denom = n_train_steps_per_rollout_batch * args.logging_steps
                avg_meta = {
                    f"avg_{key}": (sum(value) / denom)
                    for key, value in meta_data.items()
                    if value
                }
                avg_loss = avg_meta.get("avg_loss", float("nan"))
                avg_entropy = avg_meta.get("avg_entropy", float("nan"))
                avg_response_length = avg_meta.get("avg_response_length", float("nan"))

                logger.info(
                    f"step={global_step}, loss={avg_loss:.4f}, ent={avg_entropy:.4f}, response_length={avg_response_length:.4f}, time_per_step={time_per_step:.4f} seconds, estimated_time_to_completion={estimated_time_to_completion:.4f}."
                )
                log_payload = {f"train/{k}": v for k, v in avg_meta.items()}
                wandb.log(log_payload, step=global_step)  # this sets the x-axis
                meta_data = {"grad_norm": [], "loss": [], "entropy": []}

            # validation
            if (global_step+1)%args.eval_steps == 0:
                # sync policy -> vLLM
                load_policy_into_vllm_instance(policy, vllm_model)

                # validation data
                val_prompts = [system_prompt.format(question=item['question']) for item in val_data]
                val_solutions = [item['solution'] for item in val_data]
                save_path = os.path.join(
                    args.eval_save_dir,
                    f'lr_{args.lr}_B_{args.rollout_batch_size}_wd_{args.weight_decay}_step_{global_step+1}.json'
                )
                results = evaluate_vllm(
                    vllm_model, 
                    reward_fn, 
                    val_prompts, 
                    val_solutions, 
                    val_sampling_params, 
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
            if (global_step+1)%args.save_steps == 0:
                os.makedirs(args.ckpts_dir, exist_ok=True)
                torch.save(policy.state_dict(), os.path.join(args.ckpts_dir, f'sft_lr_{args.lr}_B_{args.rollout_batch_size}_wd_{args.weight_decay}_step_{global_step+1}.pth'))
                logger.info(f"Saved checkpoint at step {global_step+1}.")
            
            global_step += 1
            if global_step >= total_steps:
                return

if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)

# Run this script using python instead of torchrun since the script is single‑process and just uses two GPUs for different roles. torchrun is only needed for DDP/multi‑process training.
# nohup uv run python ./cs336_alignment/rl_grpo.py --run_name reinforce_with_baseline_lr_1e-5_B_256_wd_0.0 > ./logs/grpo.log 2>&1 &