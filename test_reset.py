import jax
import jax.numpy as jnp
from mujoco_playground import registry

# Test vectorized reset operation
env_name = "G1JoystickFlatTerrain"
env = registry.load(env_name, config=registry.get_default_config(env_name))
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

num_envs = 4
seed = 0

# Initialize environments
rng_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
states = jax.vmap(jit_reset)(rng_keys)
print(f"Initial states obs keys: {states.obs.keys()}")
print(f"Initial states obs shapes: {jax.tree.map(lambda x: x.shape, states.obs)}")

# Create some dummy actions and step
dummy_actions = jnp.zeros((num_envs, env.action_size))
next_states = jax.vmap(jit_step)(states, dummy_actions)
print(f"Next states obs shapes: {jax.tree.map(lambda x: x.shape, next_states.obs)}")

# Simulate some environments being done
dones = jnp.array([True, False, True, False])
print(f"Dones: {dones}")

# Test different reset approaches
print("\n--- Approach 1: Reset all, then select ---")
rng = jax.random.PRNGKey(42)
reset_keys = jax.random.split(rng, num_envs)
reset_states = jax.vmap(jit_reset)(reset_keys)

# Use jnp.where for selection
def select_states(reset_state, next_state, done):
    return jax.tree.map(
        lambda r, n: jnp.where(done, r, n),
        reset_state, next_state
    )

try:
    selected_states = jax.vmap(select_states)(reset_states, next_states, dones)
    print("✓ jnp.where approach works")
    print(f"Selected states obs shapes: {jax.tree.map(lambda x: x.shape, selected_states.obs)}")
except Exception as e:
    print(f"✗ jnp.where approach failed: {e}")

print("\n--- Approach 2: Conditional reset with masking ---")
def conditional_reset(state, done, reset_key):
    def reset_fn():
        return jit_reset(reset_key)
    def keep_fn():
        return state
    return jax.lax.cond(done, reset_fn, keep_fn)

try:
    reset_keys = jax.random.split(rng, num_envs)
    conditional_states = jax.vmap(conditional_reset)(next_states, dones, reset_keys)
    print("✓ Conditional reset approach works")
    print(f"Conditional states obs shapes: {jax.tree.map(lambda x: x.shape, conditional_states.obs)}")
except Exception as e:
    print(f"✗ Conditional reset approach failed: {e}")

print("\n--- Approach 3: Manual loop (current approach) ---")
try:
    updated_states = []
    reset_keys = jax.random.split(rng, int(jnp.sum(dones)))
    reset_idx = 0
    for i in range(num_envs):
        if dones[i]:
            new_state = jit_reset(reset_keys[reset_idx])
            updated_states.append(new_state)
            reset_idx += 1
        else:
            updated_states.append(jax.tree.map(lambda x: x[i], next_states))
    final_states = jax.tree.map(lambda *args: jnp.stack(args), *updated_states)
    print("✓ Manual loop approach works")
    print(f"Final states obs shapes: {jax.tree.map(lambda x: x.shape, final_states.obs)}")
except Exception as e:
    print(f"✗ Manual loop approach failed: {e}")