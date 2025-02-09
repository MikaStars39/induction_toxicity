import argparse
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformer_lens.utils import tokenize_and_concatenate

def reconstruction_test(
        model: HookedTransformer, 
        sae: SAE, batch_size: int = 1, 
        num_batches: int = 32,
):
    # Set SAE to eval mode
    sae.eval()
    dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    total_loss = 0
    with torch.no_grad():
        for i in range(num_batches):
            # Load and process a batch of tokens
            start_idx = i * batch_size
            batch_tokens = token_dataset[start_idx:start_idx + batch_size]["tokens"]
            
            # Run model and get activation cache
            _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
            
            # Get SAE features and reconstructions
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            sae_out = sae.decode(feature_acts)
            
            # Calculate reconstruction loss
            loss = torch.nn.functional.mse_loss(sae_out, cache[sae.cfg.hook_name])
            total_loss += loss.item()
            print(f"Batch {i+1}/{num_batches} - Reconstruction loss: {loss.item():.4f}")
            
            # Clean up cache
            del cache

        avg_loss = total_loss / num_batches
        print(f"\nAverage reconstruction loss: {avg_loss:.4f}")
        
        return feature_acts, sae_out

def main():
    parser = argparse.ArgumentParser(description='Run SAE reconstruction test')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                      help='Model name or path')
    parser.add_argument('--release', type=str, default="llama_scope_lxm_8x",
                      help='SAE release name')
    parser.add_argument('--sae-id', type=str, default="l31m_8x",
                      help='SAE ID')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cpu/cuda/mps). If not specified, will auto-detect')
    args = parser.parse_args()

    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() \
            else "cuda" if torch.cuda.is_available() \
                else "cpu"
    else:
        device = args.device

    model = HookedTransformer.from_pretrained(args.model, device=device)
    sae = SAE.from_pretrained(args.release, args.sae_id, device=device)[0]

    reconstruction_test(model, sae)

if __name__ == "__main__":
    main()