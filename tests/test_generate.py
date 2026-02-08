# test_generate.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from utils import generate_with_log_probs, extract_answer

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!\n")

# Test 1: Simple prompt
print("=" * 50)
print("TEST 1: Simple prompt")
print("=" * 50)
prompt = "Q: What is 7 + 5? A:"
print(f"Prompt: {prompt}\n")

response, log_prob = generate_with_log_probs(model, tokenizer, prompt, max_length=30)
print(f"Response: {response}")
print(f"Log prob: {log_prob.item():.4f}")
print(f"Extracted answer: {extract_answer(response)}\n")

# Test 2: Calculator prompt
print("=" * 50)
print("TEST 2: Calculator prompt")
print("=" * 50)
prompt = "You can use calculator(expression) to compute answers. Q: What is 23 times 45? A:"
print(f"Prompt: {prompt}\n")

response, log_prob = generate_with_log_probs(model, tokenizer, prompt, max_length=30)
print(f"Response: {response}")
print(f"Log prob: {log_prob.item():.4f}")
print(f"Extracted answer: {extract_answer(response)}\n")

# Test 3: Generate multiple responses (like GRPO does)
print("=" * 50)
print("TEST 3: Multiple responses (K=3)")
print("=" * 50)
prompt = "You can use calculator(expression) to compute answers. Q: What is 15% of 200? A:"
print(f"Prompt: {prompt}\n")

for i in range(3):
    response, log_prob = generate_with_log_probs(model, tokenizer, prompt, max_length=30)
    answer = extract_answer(response)
    used_calc = "calculator" in response.lower()
    print(f"Response {i+1}: {response}")
    print(f"  Log prob: {log_prob.item():.4f}")
    print(f"  Answer: {answer}")
    print(f"  Used calculator: {used_calc}\n")

# Test 4: Check gradients exist
print("=" * 50)
print("TEST 4: Gradient check")
print("=" * 50)
prompt = "Q: What is 10 + 20? A:"
response, log_prob = generate_with_log_probs(model, tokenizer, prompt, max_length=20)
print(f"Log prob requires grad: {log_prob.requires_grad}")
print(f"Log prob grad_fn: {log_prob.grad_fn}")

if log_prob.requires_grad:
    print("✅ Gradients are working! Ready for training.")
else:
    print("❌ No gradients! Training won't work.")