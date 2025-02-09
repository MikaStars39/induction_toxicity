import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from functools import partial
from transformers import AutoTokenizer
from src.generation import get_chat_template
from src.patch import patch
from src.utils.repeat import build_question


def benchmarking_repetition(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    chat_template: str,
    device: str,
    times: int,
    repeat_times: int,
    prefilled: int,
):
    total_result = []
    for i in range(times):
        prefix_clean, token_clean = build_question(tokenizer, repeat_times, chat_template, prefilled)
        prefix_corrupted, token_corrupted = build_question(tokenizer, repeat_times, chat_template, prefilled)
        # make sure that the clean token and the corrupted token are not the same
        while token_clean == token_corrupted:
            prefix_corrupted, token_corrupted = build_question(tokenizer, repeat_times, chat_template, prefilled)
        
        prefix_clean = tokenizer(prefix_clean, return_tensors="pt").input_ids.to(device)
        prefix_corrupted = tokenizer(prefix_corrupted, return_tensors="pt").input_ids.to(device)

        result = patch(
            prefix_clean,
            prefix_corrupted,
            int(token_clean),
            int(token_corrupted),
            model,
            device
        )
        total_result.append(result)
    
    # average the result
    average_result = sum(total_result) / len(total_result)
    
    # Get indices of top 20 values
    flattened = average_result.flatten()
    top_50_values, top_50_indices = torch.topk(flattened, 50)
    
    # Convert flat indices to 2D indices
    rows = top_50_indices // average_result.shape[1]
    cols = top_50_indices % average_result.shape[1]
    
    # Print results in requested format
    print("\nTop 10 results (layer, head, value):")
    for i in range(10):
        print(f"({rows[i]}, {cols[i]}, {top_50_values[i]:.3f})")
    
    return average_result