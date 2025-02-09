import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from functools import partial
from src.patch import patch
from src.utils.random_strings import generate_prefix

def find_induction_head(
    model: HookedTransformer,
    device: str = "cuda",
    times: int = 16,
    token: str = "",
):

    total_result = []
    for i in range(times):
        prefix = generate_prefix(model, device)
        result = patch(
            prefix["clean_tokens"], 
            prefix["corrupted_tokens"], 
            prefix["clean_verb_token"], 
            prefix["corrupted_verb_token"], 
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
    print("\nTop 20 results (layer, head, value):")
    for i in range(50):
        print(f"({rows[i]}, {cols[i]}, {top_50_values[i]:.3f})")