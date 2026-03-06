import os, sys, json, torch, math, random, time, re
from transformers import PreTrainedTokenizer, PreTrainedModel

from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, IO, BinaryIO, Tuple, Callable
from torch import Tensor, multinomial
from einops import reduce, einsum, rearrange
from jaxtyping import Float, Bool, Int

from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

def tokenizer_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
    max_len: int = None
) -> dict[str, Tensor]:
    '''
    Tokenizer a batch of prompt + response. Return a dict. Include input_ids, labels, response_mask.
    response_mask: (B, max(len of prompt + response)-1). A mask on the response token in the labels.
    input_ids: (B, max(len) - 1). The final token sliced off.
    labels: (B, max(len) - 1). The input ids without the first token.
    '''

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    assert len(prompt_strs) == len(output_strs), 'The prompt strs does not have the same length as the output strs.'

    # encode
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs]
    output_ids = [tokenizer.encode(output, add_special_tokens=False) for output in output_strs]

    # pad
    if max_len is None:
        max_len = max(len(prompt_ids[i]) + len(output_ids[i]) for i in range(len(prompt_ids)))
    token_ids = []
    response_mask = []
    for pid, oid in zip(prompt_ids, output_ids):
        p_len, o_len = len(pid), len(oid)

        # token_ids
        ids = pid + oid
        if len(ids) > max_len: # handle the case where the prompt + response is longer than the max_len.
            ids = ids[:max_len]
        ids.extend([pad_id for _ in range(max_len - len(ids))])
        token_ids.append(ids)

        # mask: handle the case where the prompt + response is longer than the max_len.
        p_len_eff = min(p_len, max_len)
        o_len_eff = max(0, min(o_len, max_len - p_len_eff))
        mask = [False for _ in range(max(p_len_eff - 1, 0))] + [True for _ in range(o_len_eff)]
        mask.extend([False for _ in range(max_len - 1 - len(mask))])
        response_mask.append(mask)

    # stack
    token_ids = torch.tensor(token_ids)
    response_mask = torch.tensor(response_mask, dtype=torch.bool)
    # make sure the response_mask is of dtype = torch.bool.

    return {
        'input_ids': token_ids[:, :-1],
        'labels': token_ids[:, 1:],
        'response_mask': response_mask
    }

def compute_entropy(
    logits: Float[Tensor, 'B T D']
) -> Float[Tensor, 'B T']:
    '''
    compute per-token entropy using a numerical stable way use log_softmax
    '''
    log_probs = torch.log_softmax(logits, dim=-1) # stable. compute p
    probs = torch.exp(log_probs)
    return - (probs * log_probs).sum(dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Int[Tensor, 'B T'],
    labels: Int[Tensor, 'B T'],
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits # get the logits. We can directly use the attributes of the PreTrainedModel class object.
    log_probs = torch.log_softmax(logits, dim=-1) # numerical stable way. (B T V)
    log_probs_at_response = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) # (B, T)
    # -1 is the dimension that we take along. index = labels.unsqueeze(-1) must have the same shape as the output (B T 1)
    result = {
        'log_probs': log_probs_at_response
    }
    if return_token_entropy:
        result['token_entropy'] = compute_entropy(logits)
    return result

def masked_normalize(
    tensor: Float[Tensor, '...'],
    mask: Float[Tensor, '...'],
    normalize_constant: float,
    dim: int | None = None
):
    '''
    sum the tensor along dimension but only at the mask==1, and then devide it by the normalize_constant
    '''
    return (tensor * mask).sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: Float[Tensor, 'B T'],
    response_mask: Float[Tensor, 'B T'],
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str,Tensor]]:
    '''
    One step of sft microbatch, including loss.backward()
    return loss and metadata (a dict which includes other statistics that we want to log, for example, the average response length, and the average token entropy at the responses).

    The loss is averaged over the batch size and the gradient accumulation steps, as well as the additional normalization constant.
    '''
    # compute the loss
    batch_size = policy_log_probs.shape[0]
    loss = -1.0 * masked_normalize( # negative log likelihood loss
        policy_log_probs,
        response_mask,
        normalize_constant * gradient_accumulation_steps * batch_size, # include the grad accum and average over batch
    )
    loss.backward()
    metadata = {}
    return loss, metadata

# load model with vLLM
def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.6
):
    '''
    start the inference process. Here we use vLLM to hold a model on a GPU separate from the policy.
    '''

    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).

    world_size_patch = patch('torch.distributed.get_world_size', return_value=1)
    # RZ: torch.distributed.get_world_size → always returns 1 (forces vLLM to think it’s single‑GPU).
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    # RZ: vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling → no‑op (disables a profiling check that can fail in this setup).

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16, # RZ: This sets the model weights and computation to use bfloat16 (BF16). It’s mainly about parameters + activations, not just the output. 
            enable_prefix_caching=True, # RZ: vLLM will cache KV states for shared prompt prefixes across requests. If you reuse the same prefix many times (for example, the system prompt), generation is faster. It uses extra memory to store these cached prefixes.
            gpu_memory_utilization=gpu_memory_utilization, # RZ: A fraction of GPU memory that vLLM is allowed to reserve for the KV cache. Higher values increase throughput/concurrency but consume more memory. Lowering it reduces memory usage (at the cost of fewer concurrent tokens/requests).
        )

def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    llm: LLM,
):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict() # get all the parameters of the policy model. A dict. name to tensors.
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model # Reach the vllm internal to grad the underlying model.
    llm_model.load_weights(state_dict.items()) # load the parameters into the vllm model.