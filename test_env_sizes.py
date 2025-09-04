#!/usr/bin/env python3
"""Quick test to verify env.observation_size vs actual flattened obs size"""

import jax
import jax.numpy as jnp
from mujoco_playground import registry

def flatten_obs(obs):
    if isinstance(obs, dict):
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1)
    return obs

# Test just CheetahRun quickly
env_name = "CheetahRun"
print(f"Testing {env_name}...")

env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)

print(f"env.observation_size: {env.observation_size}")
print(f"env.action_size: {env.action_size}")