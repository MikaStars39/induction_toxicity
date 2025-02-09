import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from functools import partial

def logits_to_ave_logit_diff(
    logits: torch.Tensor,
    answer_tokens: torch.Tensor,
    per_prompt: bool = False
) -> torch.Tensor:
    '''
    Returns logit difference between the correct and incorrect answer.
    answer_tokens should contain [correct_token, incorrect_token]
    '''
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def ioi_metric(
    logits: torch.Tensor, 
    answer_tokens: torch.Tensor,
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> float:
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def patch(
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    clean_verb_token: int,
    corrupted_verb_token: int,
    model: HookedTransformer, 
    device: str,
):

    answer_tokens = torch.tensor([[clean_verb_token, corrupted_verb_token]], device=device)

    # check if the clean tokens and corrupted tokens are the same length
    min_len = min(clean_tokens.shape[1], corrupted_tokens.shape[1])
    clean_tokens = clean_tokens[:, :min_len]
    corrupted_tokens = corrupted_tokens[:, :min_len]

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)

    one_head_act_patch_result = patching.get_act_patch_attn_head_out_all_pos(
        model,
        corrupted_tokens,
        clean_cache,
        partial(
        ioi_metric,
        answer_tokens=answer_tokens,
        clean_logit_diff=clean_logit_diff,
        corrupted_logit_diff=corrupted_logit_diff
        )
    )

    return one_head_act_patch_result