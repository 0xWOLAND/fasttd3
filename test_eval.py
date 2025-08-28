#!/usr/bin/env python3
import time, jax, jax.numpy as jnp
from jax import lax
from functools import partial
from brax import envs

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compilation_cache")

# --- Config ---
backend = "mjx"       # try "positional" or "generalized"
batch_size = 2
steps_per_call = 128
repeats = 3

# --- Env ---
env = envs.create("humanoid", batch_size=batch_size, backend=backend)
obs_dim, act_dim = env.observation_size, env.action_size

def sync(x): jax.block_until_ready(x.reward if hasattr(x, "reward") else x)

# --- Baseline (raw calls) ---
rng = jax.random.PRNGKey(0)
t0 = time.time(); state = env.reset(rng); sync(state)
print("reset (compile):", time.time()-t0)

actions = jnp.zeros((batch_size, act_dim))
t0 = time.time(); state = env.step(state, actions); sync(state)
print("step  (compile):", time.time()-t0)

# steady raw
for i in range(repeats):
    t0 = time.time(); state = env.step(state, actions); sync(state)
    print(f"step raw {i}:", time.time()-t0)

# --- JIT reset/step ---
reset_jit = jax.jit(env.reset)
step_jit  = jax.jit(env.step, donate_argnums=(0,))

t0 = time.time(); state = reset_jit(rng); sync(state)
print("reset_jit (compile):", time.time()-t0)

t0 = time.time(); state = step_jit(state, actions); sync(state)
print("step_jit  (compile):", time.time()-t0)

for i in range(repeats):
    t0 = time.time(); state = step_jit(state, actions); sync(state)
    print(f"step_jit {i}:", time.time()-t0)

for i in range(repeats):
    t0 = time.time(); state = reset_jit(rng); sync(state)
    print(f"reset_jit {i}:", time.time()-t0)

# --- JIT multi-step scan ---
@partial(jax.jit, donate_argnums=(0,))
def step_n(state, acts_seq):
    def body(s, a): s = env.step(s, a); return s, s.reward
    return lax.scan(body, state, acts_seq)

acts_seq = jnp.zeros((steps_per_call, batch_size, act_dim))
t0 = time.time(); (state, _)= step_n(state, acts_seq); sync(state)
print("step_n (compile):", time.time()-t0)

for i in range(repeats):
    t0 = time.time(); (state, r)= step_n(state, acts_seq); sync(state)
    print(f"step_n {i} total:", time.time()-t0,
          "per-step:", (time.time()-t0)/steps_per_call)
