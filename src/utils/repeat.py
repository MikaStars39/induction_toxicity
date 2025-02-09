from transformers import AutoTokenizer
import random
import torch
from src.patch import patch

tokens = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "apple", "car", "house", "tree", "fire",
    "%", ",", "."
]

def build_question(
    tokenizer: AutoTokenizer,
    repeat_times: int,
    chat_template: str,
    prefilled: int = 3,
) -> str:
    template = (
        "Example: Please repeat this word: Apple in 10 times. Assistant: Apple Apple Apple Apple Apple Apple Apple Apple Apple Apple.\n"
        "Now it's your turn.\n"
        "Please repeat this word: {} in {} times. You cannot simplify this process.\n"
        "You need to actually repeat the word. Don't worry about the context length, just repeat the word.\n"
        "There should be no other words except the token: {}."
    )
    
    # randomly select a token from the vocabulary
    token = tokens[random.randint(0, len(tokens) - 1)]
    return chat_template.format(template.format(token, repeat_times, token)) + token * prefilled, tokenizer.encode(token)[0]


def counting_that_token(
    inputs: torch.Tensor,
    token: int,
) -> int:
    return (inputs == token).sum().item()

