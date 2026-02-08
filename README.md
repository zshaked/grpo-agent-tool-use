# GRPO for Agent Tool Use

Applying Group Relative Policy Optimization (GRPO) to teach agents when and how to use tools.

## What is GRPO?

GRPO improves on standard policy gradient by computing advantages relative to a group of responses:
1. Generate K responses to the same prompt
2. Score each response (tool use + correctness)
3. Compute advantage = reward - group_mean
4. Update model to increase probability of above-average responses

## Reward Design

The reward function encourages both tool use AND correctness:

| Scenario | Reward |
|----------|--------|
| Used calculator + correct | 1.0 |
| Used calculator + wrong | 0.3 |
| No calculator + correct | 0.5 |
| No calculator + wrong | 0.0 |

## Why This Matters for Agent Safety

This directly relates to training safe, capable agents:
- **Tool use**: Teaching agents when to invoke tools
- **Reward shaping**: Balancing multiple objectives
- **Verification**: Using verifiable outcomes to improve behavior

## Usage
```bash
pip install -r requirements.txt
python train_agent.py
```

## Files

- `train_agent.py` - Main training script with GRPO loop
- `utils.py` - Helper functions (generation, advantages)
- `data.py` - Calculator problem generation

## Results

Training demonstrates the GRPO algorithm for agent tool use. With more compute/data, the agent would learn to consistently invoke the calculator.