#!/usr/bin/env python3
"""Test vectorized environments in Brax - shortest possible way"""

import brax.envs as envs
import jax
import jax.numpy as jnp

# Create vectorized environment using built-in API
env = envs.create(
    env_name="humanoid",
    episode_length=1000,
    action_repeat=1,
    auto_reset=True,
    batch_size=8,  # This automatically applies VmapWrapper
    backend="mjx"
)

# Initialize
rng = jax.random.PRNGKey(0)
rng, reset_key = jax.random.split(rng)

# Reset environment batch
state = env.reset(reset_key)
print(f"State shape: obs={state.obs.shape}, reward={state.reward.shape}, done={state.done.shape}")

# Step with random actions
for i in range(5):
    rng, action_key = jax.random.split(rng)
    actions = jax.random.uniform(action_key, (8, env.action_size), minval=-1, maxval=1)
    state = env.step(state, actions)
    print(f"Step {i+1}: rewards={state.reward}, dones={state.done}")

# Verify it's actually vectorized by checking different environments have different states
print(f"\nObs differences between env 0 and 7:")
print(f"Obs[0][:5] = {state.obs[0][:5]}")
print(f"Obs[7][:5] = {state.obs[7][:5]}")
print(f"Are they different? {not jnp.allclose(state.obs[0], state.obs[7])}")

print("✓ Vectorized Brax environments working!")