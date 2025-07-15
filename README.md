# `fasttd3` 

A pure JAX port of [FastTD3](https://github.com/younggyoseo/FastTD3) - a high-performance variant of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

This implementation aims to replicate the FastTD3 paper's approach using JAX/Flax with distributional critics and vectorized training.

## Usage

```bash
uv run train.py
```

## Features

- Pure JAX/Flax implementation for fast training
- Distributional critics with 51 atoms
- Vectorized training with 128 parallel environments
- Replay buffer with n-step returns

## Status

Parameter tuning is still ongoing.

## Reference

Based on the FastTD3 paper: "FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control" and the original TD3 paper: "Addressing Function Approximation Error in Actor-Critic Methods"

## todo
[] checkpointing