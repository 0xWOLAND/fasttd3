import mujoco_playground
from mujoco_playground import registry
import numpy as np
import pickle
from td3.td3 import Actor
import jax
import jax.numpy as jnp

# Helper to flatten observations
def flatten_obs(obs):
    if isinstance(obs, dict):
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1)
    return obs

# Load saved model
with open("trained_td3.pkl", "rb") as f:
    data = pickle.load(f)
    actor_params = data["actor_params"]
    obs_dim = data["obs_dim"]
    act_dim = data["act_dim"]
    max_action = data["max_action"]

actor = Actor(obs_dim, act_dim, max_action, hidden_dim=256)
env = registry.load("G1JoystickFlatTerrain", config=registry.get_default_config("G1JoystickFlatTerrain"))
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)

for episode in range(3):
    rng = jax.random.PRNGKey(episode)
    state = jit_reset(rng)
    episode_reward = 0
    
    print(f"Episode {episode + 1}")
    
    while not state.done:
        flat_obs = flatten_obs(state.obs)
        obs_jax = jnp.array(flat_obs[None])
        
        action = actor.apply(actor_params, obs_jax)[0]
        action = np.asarray(action)
        
        state = jit_step(state, action)
        episode_reward += state.reward
    
    print(f"  Reward: {episode_reward:.2f}")