from fire import Fire
from omegaconf import OmegaConf
import torch
import random

from src.generation import get_chat_template

def load_model_and_tokenizer(
    model_name: str,
    device: str,
    task_name: str,
):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "repeat" in task_name:
        if_hooked = False
    else:
        if_hooked = True
        
    if if_hooked:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            torch_dtype=torch.bfloat16
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    chat_template = get_chat_template(model_name)
    
    return tokenizer, model, chat_template

def main(
    task_name: str,
    model_name: str,
    token: str,
    seed: int = 31,
    config_path: str = "config.yaml",
):
    config = OmegaConf.load(config_path)

    from huggingface_hub import login
    login(token=token)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    tokenizer, model, chat_template = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        task_name=task_name,
    )
    config.chat_template = chat_template
    config.model_name = model_name

    if "induction" in task_name:
        from src.benchmark.induction_head import find_induction_head
        find_induction_head(
            model=model,
            device=device,
            times=config.times,
        )
    
    if "repeat" in task_name:
        from src.generation import load_and_generate_llm
        load_and_generate_llm(
            model=model,
            tokenizer=tokenizer,
            device=device,
            **config,
        )
    
    if "patch" in task_name:
        from src.benchmark.repeat import benchmarking_repetition
        benchmarking_repetition(
            model=model,
            tokenizer=tokenizer,
            device=device,
            **config,
        )

if __name__ == "__main__":
    Fire(main)
