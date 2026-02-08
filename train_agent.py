from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from data import generate_calculator_problems
from utils import extract_answer, generate_with_log_probs, compute_advantages

def compute_tool_use_rewards(responses, correct_answer):
    """
    Your reward logic here.
    
    Think through:
    - How to check if tool was used
    - How to check correctness
    - What reward for each combination
    """
    rewards = []

    for response in responses:
        used_calculator = "calculator" in response.lower()
        answer = extract_answer(response)
        correct = (answer == correct_answer)
        if correct and used_calculator:
            rewards.append(1.0)
        elif not correct and used_calculator:
            rewards.append(0.3)
        elif correct and not used_calculator:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return np.array(rewards)



def train_agent_grpo(model, tokenizer, problems, num_epochs=3, k=3, lr=5e-5):
    """Train agent to use calculator tool with GRPO"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training agent on {len(problems)} problems")
    print(f"Epochs: {num_epochs}, Responses per problem: {k}\n")

    for epoch in range(num_epochs):
        epoch_correct = 0
        epoch_tool_use = 0
        epoch_total = 0
        
        for i, problem in enumerate(problems):
            question = problem['question']
            correct_answer = problem['answer']
            
            # Prompt tells model about calculator tool
            prompt = f"You can use calculator(expression) to compute answers. Q: {question} A:"
            
            # Generate k responses
            responses = []
            log_probs = []
            for _ in range(k):
                response, log_prob = generate_with_log_probs(
                    model, tokenizer, prompt, max_length=25
                )
                responses.append(response)
                log_probs.append(log_prob)
            
            # Use tool-aware reward function
            rewards = compute_tool_use_rewards(responses, correct_answer)
            advantages = compute_advantages(rewards)
            
            # Policy gradient
            loss = sum(-adv * lp for adv, lp in zip(advantages, log_probs)) / len(log_probs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track stats
            for response in responses:
                if extract_answer(response) == correct_answer:
                    epoch_correct += 1
                if "calculator" in response.lower():
                    epoch_tool_use += 1
            epoch_total += len(responses)
            
            del loss, log_probs
            
            # Print progress every 5 problems
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Problem {i+1}/{len(problems)}")
                print(f"  Tool use rate: {epoch_tool_use/epoch_total:.1%}")
                print(f"  Accuracy: {epoch_correct/epoch_total:.1%}\n")
        
        # Epoch summary
        print(f"{'='*60}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*60}")
        print(f"Tool use: {epoch_tool_use}/{epoch_total} ({epoch_tool_use/epoch_total:.1%})")
        print(f"Accuracy: {epoch_correct}/{epoch_total} ({epoch_correct/epoch_total:.1%})\n")
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("GRPO for Agent Tool Use")
    print("="*60 + "\n")
    
    print("Loading DistilGPT-2...")
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Generating calculator problems...")
    problems = generate_calculator_problems(n=10)  # Start with 10
    
    print("Training...\n")
    model = train_agent_grpo(model, tokenizer, problems, num_epochs=2, k=3)
    
    print("Done!")
        