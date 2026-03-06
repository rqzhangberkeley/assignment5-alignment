import os, sys, json, torch, math, random, time, re
from transformers import PreTrainedTokenizer, PreTrainedModel

from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, IO, BinaryIO, Tuple, Callable, Literal
from torch import Tensor, multinomial
from einops import reduce, einsum, rearrange 
from jaxtyping import Float, Bool, Int

from cs336_alignment.sft_utils import compute_entropy

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalized_by_std: bool
) -> tuple[Tensor, Tensor, dict[str,float]]:
    '''
    compute the group normalized reward (group advantage).
    Support the GRPO style advantage and the Dr.GRPO style advantage.
    len(rollout_responses) = len(repeated_ground_truth) = group_size * number of prompts per rollout batch.
    return:
        advantage: (rollout_batch_size,)
        raw_reward: (rollout_batch_size,)
        meta_data: which can contains mean, std, max, min of rewards. Default to be dict().
    '''

    # compute reward
    rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(response, ground_truth)
        rewards.append(reward['reward']) # A single reward for each response.

    # compute the group advantage
    n_prompts = len(rewards) // group_size
    assert len(rewards) % group_size == 0, f'The rollout batch size must be a multiple of the group size. Get rollout_batch_size = {len(rewards)} and group_size = {group_size}.'
    rewards = torch.tensor(rewards).reshape((n_prompts, group_size))
    advantage = rewards - rewards.mean(dim=-1, keepdim=True)
    if normalized_by_std:
        advantage = advantage / (rewards.std(dim=-1, keepdim=True) + advantage_eps)

    # meta_data
    meta_data = dict()
    return advantage.reshape(-1), rewards.reshape(-1), meta_data

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Float[Tensor, 'B 1'],
    policy_log_probs: Float[Tensor, 'B T'],
) -> Float[Tensor, 'B T']:
    '''
    Naive policy gradient loss. The loss is per token.
    '''
    return -1.0 * raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: Float[Tensor, 'B 1'],
    policy_log_probs: Float[Tensor, 'B T'],
    old_log_probs: Float[Tensor, 'B T'],
    cliprange: float
) -> tuple[Tensor, dict[str, Tensor]]:
    '''
    GRPO clip loss. Per token.
    meta_data: save whether each token was clipped or not.
    '''
    policy_ratios = torch.exp(policy_log_probs - old_log_probs)
    policy_ratios_clipped = torch.clamp(policy_ratios, min=1.0-cliprange, max=1.0+cliprange)

    policy_ratios_adv = policy_ratios * advantages # broadcast
    policy_ratios_clipped_adv = policy_ratios_clipped * advantages # broadcast

    clipped = policy_ratios_adv > policy_ratios_clipped_adv # whether this token is clipped

    return -1.0 * torch.minimum(policy_ratios_adv, policy_ratios_clipped_adv), dict(clipped=clipped)

def compute_policy_gradient_loss(
    policy_log_probs: Float[Tensor, 'B T'],
    loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
    raw_rewards: Float[Tensor, 'B T'] | None = None,
    advantages: Float[Tensor, 'B T'] | None = None,
    old_log_probs: Float[Tensor, 'B T'] | None = None,
    clip_range: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    '''
    Compute the PG loss. return loss, meta_data
    '''
    assert loss_type in ['no_baseline', 'reinforce_with_baseline', 'grpo_clip'], f'Invalid loss type: {loss_type}.'
    if loss_type == 'no_baseline':
        assert raw_rewards is not None, 'Raw rewards are required for no baseline loss.'
        loss, meta_data = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None, 'Advantages are required for reinforce with baseline loss.'
        loss, meta_data = compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == 'grpo_clip':
        assert advantages is not None, 'Advantages are required for GRPO clip loss.'
        assert old_log_probs is not None, 'Old log probabilities are required for GRPO clip loss.'
        assert clip_range is not None, 'Clip range is required for GRPO clip loss.'
        loss, meta_data = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    return loss, meta_data

def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None
):
    mask = mask.to(dtype=tensor.dtype) # RZ: We cast mask to the same dtype as tensor so the multiply and sum behave consistently (no implicit bool→int conversions or dtype mismatches). It ensures (tensor * mask) and mask.sum(...) produce floating outputs suitable for division and NaN propagation when the mask is all zeros.
    masked_sum = (tensor * mask).sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    return masked_sum / mask_sum

def grpo_microbatch_train_step(
    policy_log_probs: Float[Tensor, 'B T'],
    response_mask: Float[Tensor, 'B T'],
    gradient_accumulation_steps: int,
    loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
    raw_rewards: Float[Tensor, 'B T'] | None = None,
    advantages: Float[Tensor, 'B T'] | None = None,
    old_log_probs: Float[Tensor, 'B T'] | None = None,
    cliprange: float | None = None
):
    '''
    One microbatch of GRPO loss. Include loss.backward()
    '''
    # check arguments
    if loss_type == 'no_baseline':
        assert raw_rewards is not None, 'Raw rewards are required for no baseline loss.'
    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None, 'Advantages are required for reinforce with baseline loss.'
    elif loss_type == 'grpo_clip':
        assert advantages is not None, 'Advantages are required for GRPO clip loss.'
        assert old_log_probs is not None, 'Old log probabilities are required for GRPO clip loss.'
        assert cliprange is not None, 'Clip range is required for GRPO clip loss.'
    
    # compute token-wise loss
    loss, meta_data = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)

    # masked mean the loss
    loss = masked_mean(loss, response_mask)

    # scale the loss
    loss = loss / gradient_accumulation_steps

    # backward
    loss.backward()

    # process the meta_data
    if not meta_data:
        meta_data = {}
    # average the token-wise meta_data.
    if 'clipped' in meta_data:
        meta_data['clipped'] = masked_mean(meta_data['clipped'].float(), response_mask).item() / gradient_accumulation_steps
    meta_data['response_length'] = masked_mean(response_mask.float(), response_mask).item() / gradient_accumulation_steps
    return loss, meta_data

def collate_fn_grpo(batch, system_prompt, tokenizer):
    # batch is a list of samples from your dataset.
    prompts = [system_prompt.format(question=x["question"]) for x in batch]
    outputs = [x["solution"] for x in batch]
    return prompts, outputs