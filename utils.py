import re
import numpy as np
import torch

def extract_answer(response):
    """Extract the last number from response (usually the answer)"""
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        return numbers[-1]
    return ""

def compute_advantages(rewards):
    """GRPO: advantages relative to group mean"""
    mean = np.mean(rewards)
    return rewards - mean

def generate_with_log_probs(model, tokenizer, prompt, max_length=20, temperature=0.7):
    """
    Generate response while tracking log probabilities.
    Copy this from your grpo-arithmetic project!
    """
    generated = tokenizer.encode(prompt, return_tensors='pt')
    log_probs = []

    for _ in range(max_length - len(generated[0])):
        outputs = model(generated)
        logits = outputs.logits[:,-1,:]
        probs = torch.softmax(logits/temperature, dim=-1)

        next_token_id = torch.multinomial(probs.detach(), num_samples=1)
        log_prob = torch.log(probs[0,next_token_id])
        log_probs.append(log_prob)

        generated = torch.cat([generated, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0, requires_grad=True)

    return response, total_log_prob

