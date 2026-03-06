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
from cs336_alignment.rl_utils import masked_mean

# reward and evaluation functions
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm

from vllm import LLM, SamplingParams

os.environ['VLLM_DISABLE_TQDM'] = '1'

# helper functions from the SFT part.
from cs336_alignment.sft import setup_logger, load_jsonl

def collate_fn_ei(batch, system_prompt):
    # batch is a list of samples from your dataset.
    prompts = [system_prompt.format(question=x["question"]) for x in batch]
    outputs = [x["solution"] for x in batch] # Use the CoT
    return prompts, outputs

def build_parser():
    parser = argparse.ArgumentParser(
        description="Training a model using Expert Iteration. This is the main training script."
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
    parser.add_argument('--eval_save_dir', type = str, default = '/home/ruiqizhang/CS336/assignment5-alignment/results/expert_iteration')
    parser.add_argument('--ckpts_dir', type = str, default = "/home/ruiqizhang/CS336/assignment5-alignment/ckpts")

    # Inference.
    parser.add_argument('--sampling_min_tokens', type = int, default = 4, help = "Min sequence length that we can generate at training and validation time. This is to avoid the empty string.")
    parser.add_argument('--sampling_max_tokens', type = int, default = 1024, help = "Max sequence length that we can generate at training and validation time")
    parser.add_argument('--sampling_temperature', type = float, default = 1.0)
    parser.add_argument('--gpu_memory_utilization', type = float, default = 0.85)

    # training
    parser.add_argument('--max_expert_iteration_steps', type = int, default = 5)
    parser.add_argument('--num_prompts_per_step', type = int, default = 512)
    parser.add_argument('--group_size', type = int, default = 8, help = 'The number of responses we generate for each expert iteration step.')
    parser.add_argument('--train_batch_size', type = int, default = 32, help = 'The number of responses we train for each training step.')
    parser.add_argument('--micro_train_batch_size', type = int, default = 4, help = 'The number of responses we train for each micro training step. We can compute the gradient accumulation steps by dividing the train batch size by this value.')
    

    # optimizer
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--max_grad_norm', type = float, default = 1.0)

    # log
    parser.add_argument('--logging_steps', type = int, default = 1)
    parser.add_argument('--save_steps', type = int, default = 10000)
    parser.add_argument('--run_name', type = str, default = 'expert_iteration_run_test')

    # eval
    parser.add_argument('--eval_steps', type = int, default = 1, help = 'The number of training steps between each evaluation. This is not the number of expert iteration steps between each evaluation.')
    parser.add_argument('--reward', type = str, default = 'drgrpo')

    return parser

def main(args):

    # init
    logger = setup_logger('Train a small model on math dataset using Expert Iteration with Verifieable Rewards.......')
    wandb.init(project="cs336-expert-iteration", name=args.run_name)

    # assert
    assert args.reward == 'drgrpo', f'Invalid reward function at validation: {args.reward}.'
    assert args.train_batch_size % args.micro_train_batch_size == 0, (
        'train_batch_size must be divisible by micro_train_batch_size'
    )
    gradient_accumulation_steps = args.train_batch_size // args.micro_train_batch_size

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
        batch_size=args.num_prompts_per_step,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_ei(batch, system_prompt),
    )
    # RZ Notes: Here batch size is the number of prompts that we use for each rollout batch. 
    # Add system prompt in the data collator. The data collator will provide the solutions.

    # total steps
    total_steps = args.max_expert_iteration_steps
    
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
    training_steps = 0 # the number of steps that we have completed
    start_time = time.time()

    # accumulate metrics across logging_steps
    meta_data = {"grad_norm": [], "loss": [], "entropy": [], "response_length": []}

    for expert_iteration_step in range(args.max_expert_iteration_steps):
        for prompts, solutions in train_loader:

            # generate responses. Currently, we do not get the log-P directly from the vllm.
            load_policy_into_vllm_instance(policy, vllm_model)
            outputs = vllm_model.generate(prompts, train_sampling_params)
            outputs_text = [output.outputs[i].text for output in outputs for i in range(args.group_size)] # pay attention to the order
            repeated_prompts = [p for p in prompts for _ in range(args.group_size)]
            repeated_solutions = [s for s in solutions for _ in range(args.group_size)]

            # compute the group normalized rewards
            _, raw_rewards, _ = compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=outputs_text,
                repeated_ground_truths=repeated_solutions,
                group_size=args.group_size,
                advantage_eps=1e-6,
                normalized_by_std=False
            )

            # select the correct responses for training.
            correct_idx = raw_rewards > 0
            correct_prompts = [repeated_prompts[i] for i in range(len(repeated_prompts)) if correct_idx[i]]
            correct_generated_solutions = [outputs_text[i] for i in range(len(outputs_text)) if correct_idx[i]]
            num_correct_responses = len(correct_prompts)
            if num_correct_responses == 0:
                print(f'No correct responses found. Skipping this step.')
                continue
            print(f'Step {expert_iteration_step} of {args.max_expert_iteration_steps}: selecting the correct responses for training...')
            print(f'Total {len(prompts)} prompts and {len(repeated_solutions)} solutions. Correct responses: {num_correct_responses}')


            # shuffle the correct responses
            paired = list(zip(correct_prompts, correct_generated_solutions))
            random.shuffle(paired)
            correct_prompts, correct_generated_solutions = zip(*paired)
            correct_prompts, correct_generated_solutions = list(correct_prompts), list(correct_generated_solutions)

            # SFT the policy model on the correct responses with train_batch_size
            # One training step.
            for i in range(0, num_correct_responses, args.train_batch_size):
                batch_prompts = correct_prompts[i:i+args.train_batch_size]
                batch_generated_solutions = correct_generated_solutions[i:i+args.train_batch_size]
                batch_size = len(batch_prompts) # The last training batch may have less data than the train_batch_size.
                effective_accum = math.ceil(batch_size / args.micro_train_batch_size) # RZ: For other batches except the last one, this should equals to grad_accum. This is only for the last batch where the batch size can be smaller than train_batch_size.
                # And the batch size for each micro training step is args.micro_train_batch_size.

                for micro_start in range(0, batch_size, args.micro_train_batch_size):
                    micro_prompts = batch_prompts[micro_start:micro_start + args.micro_train_batch_size]
                    micro_generated_solutions = batch_generated_solutions[micro_start:micro_start + args.micro_train_batch_size]

                    # tokenize the prompts and the solutions
                    batch_rollout_batch = tokenizer_prompt_and_output(prompt_strs=micro_prompts, output_strs=micro_generated_solutions, tokenizer=tokenizer,max_len=args.sampling_max_tokens)
                    micro_batch_input_ids = batch_rollout_batch['input_ids'].to(args.policy_device)
                    micro_batch_labels = batch_rollout_batch["labels"].to(args.policy_device)
                    micro_batch_response_mask = batch_rollout_batch["response_mask"].to(args.policy_device) # bool

                    # compute the log P and the entropy
                    log_probs_result = get_response_log_probs(
                        model=policy,
                        input_ids=micro_batch_input_ids,
                        labels=micro_batch_labels,
                        return_token_entropy=True,
                    )
                    micro_batch_log_probs = log_probs_result['log_probs']  # B, T
                    micro_batch_token_entropy = log_probs_result['token_entropy']  # B, T

                    # compute the loss and the entropy. Averaged over gradient accumulation steps. So we devide it by the gradient accumulation steps.
                    loss = -1.0 * masked_mean(micro_batch_log_probs, micro_batch_response_mask) / effective_accum # scalar
                    token_entropy = masked_mean(micro_batch_token_entropy, micro_batch_response_mask) / effective_accum # scalar
                    loss.backward() # On each loss.backward(), PyTorch adds the new gradients into the existing p.grad buffer (i.e., p.grad += new_grad). It does not overwrite unless you clear it.
                    meta_data['loss'].append(loss.item())
                    meta_data['entropy'].append(token_entropy.item())
                    meta_data['response_length'].append(masked_mean(micro_batch_response_mask, micro_batch_response_mask).item() / effective_accum)

                # Do one step of gradient update.
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm) # clip the gradient
                meta_data["grad_norm"].append(grad_norm.item()) # One number every train step.
                optimizer.step()
                optimizer.zero_grad()

                # update the training steps
                training_steps += 1

                # logging.
                if training_steps%args.logging_steps == 0:
                    current_time = time.time()
                    time_per_step = (current_time - start_time) / (training_steps + 1)
                    estimated_time_to_completion = time_per_step * (total_steps - training_steps - 1)

                    denom = max(len(meta_data['loss']), 1)
                    avg_loss = sum(meta_data['loss']) / denom
                    avg_entropy = sum(meta_data['entropy']) / max(len(meta_data['entropy']), 1)
                    avg_response_length = sum(meta_data['response_length']) / max(len(meta_data['response_length']), 1)
                    avg_grad_norm = sum(meta_data['grad_norm']) / denom
                    logger.info(
                        f"step={training_steps}, loss={avg_loss:.4f}, ent={avg_entropy:.4f}, response_length={avg_response_length:.4f}, time_per_step={time_per_step:.4f} seconds, estimated_time_to_completion={estimated_time_to_completion:.4f}."
                    )
                    wandb.log(
                        {f"train/loss": avg_loss, f"train/entropy": avg_entropy, f"train/response_length": avg_response_length, f"train/grad_norm": avg_grad_norm},
                        step=training_steps
                    )

                    # reset the meta_data
                    meta_data = {"grad_norm": [], "loss": [], "entropy": [], "response_length": []}

            # validation
            # Every expert iteration step, we do one validation.
            # sync policy -> vLLM
            load_policy_into_vllm_instance(policy, vllm_model)

            # validation data
            val_prompts = [system_prompt.format(question=item['question']) for item in val_data]
            val_solutions = [item['solution'] for item in val_data]
            save_path = os.path.join(
                args.eval_save_dir,
                f'lr_{args.lr}_B_{args.num_prompts_per_step}_Btrain_{args.train_batch_size}_wd_{args.weight_decay}_step_{expert_iteration_step+1}.json'
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
                f"validation:step={expert_iteration_step} avg_reward={avg_reward:.4f} avg_format_reward={avg_format_reward:.4f}."
            )
            wandb.log(
                {
                    "val/avg_reward": avg_reward,
                    "val/avg_format_reward": avg_format_reward,
                }, 
                step=expert_iteration_step   # this sets the x-axis
            )

            # save the ckpts.
            if (expert_iteration_step + 1) % args.save_steps == 0:
                os.makedirs(args.ckpts_dir, exist_ok=True)
                torch.save(policy.state_dict(), os.path.join(args.ckpts_dir, f'ei_lr_{args.lr}_B_{args.num_prompts_per_step}_Btrain_{args.train_batch_size}_wd_{args.weight_decay}_step_{expert_iteration_step+1}.pth'))
                logger.info(f"Saved checkpoint at step {expert_iteration_step+1}.")

if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)

# Run this script using python instead of torchrun since the script is single‑process and just uses two GPUs for different roles. torchrun is only needed for DDP/multi‑process training.
# nohup uv run python ./cs336_alignment/expert_iteration.py --run_name expert_iteration_lr_1e-5_B_256_Btrain_32_wd_0.0 > ./logs/expert_iteration.log 2>&1 &