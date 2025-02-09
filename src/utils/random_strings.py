import torch
from functools import partial

def generate_prefix(model, device, k=20):
    # Generate a random string of length k
    import random
    import string
    # Include all printable characters except whitespace
    all_chars = string.digits + string.ascii_letters + string.punctuation
    
    # Choose a random position (not first or last) to test
    pos = random.randint(1, k-2)
    
    # First choose the first_token and remove it from available chars
    first_token = random.choice(all_chars)
    available_chars = all_chars.replace(first_token, '')
    
    # Generate rest of string without the first_token
    random_string = ''.join(random.choices(available_chars, k=k))
    
    # Insert first_token at pos
    random_string = random_string[:pos] + first_token + random_string[pos:]
    
    # Get the tokens we need
    second_token = random_string[pos+1]
    
    # Choose another random char that's not the second token for corruption
    available_chars = available_chars.replace(second_token, '')
    another_token = random.choice(available_chars)
    
    # Create clean and corrupted strings
    clean_prompt = random_string
    corrupted_prompt = random_string[:pos+1] + another_token + random_string[pos+2:]

    # Convert to tokens
    tokenizer = model.tokenizer
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(device)
    corrupted_tokens = tokenizer(corrupted_prompt, return_tensors="pt").input_ids.to(device)

    # Get token IDs for the answers
    clean_verb_token = tokenizer(second_token, add_special_tokens=False).input_ids[0]
    corrupted_verb_token = tokenizer(another_token, add_special_tokens=False).input_ids[0]

    qwen_chat_template = "<|im_start|>user\nHere is a string: {}\
        \n The character after {} is ?<|im_end|>\n<|im_start|>assistant\n The answer is"
    
    clean_prompt = qwen_chat_template.format(clean_prompt, first_token)
    corrupted_prompt = qwen_chat_template.format(corrupted_prompt, first_token)

    return {
        "clean_tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
        "clean_verb_token": clean_verb_token,
        "corrupted_verb_token": corrupted_verb_token
    }
