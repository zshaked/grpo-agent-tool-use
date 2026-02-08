# GRPO for Agent Tool Use

Applying Group Relative Policy Optimization (GRPO) to teach agents when to use tools.

## Overview

This project demonstrates how GRPO can train agents to:
1. Recognize when a tool is needed
2. Invoke the tool appropriately
3. Produce correct answers

## What is GRPO?

GRPO improves on standard policy gradient by comparing responses within a group:

1. Generate K responses to the same prompt
2. Score each response (tool use + correctness)
3. Compute advantage = reward - group_mean
4. Update model: increase probability of above-average responses

## Reward Design

| Scenario | Reward |
|----------|--------|
| Used calculator + correct | 1.0 |
| Used calculator + wrong | 0.3 |
| No calculator + correct | 0.5 |
| No calculator + wrong | 0.0 |

This reward shaping encourages tool use as the preferred strategy.

## Usage
```bash