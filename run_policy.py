import gymnasium as gym
import numpy as np
import pickle
from td3.td3 import Actor
import jax
import jax.numpy as jnp

# Load saved model
with open("halfcheetah_trained_td3.pkl", "rb") as f:
    data = pickle.load(f)
    actor_params = data["actor_params"]
    obs_dim = data["obs_dim"]
    act_dim = data["act_dim"]
    max_action = data["max_action"]

actor = Actor(obs_dim, act_dim, max_action, hidden_dim=256)
env = gym.make("HalfCheetah-v5", render_mode="human")

for episode in range(3):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    
    print(f"Episode {episode + 1}")
    
    while not done:
        obs_jax = jnp.array(obs[None])
        
        action = actor.apply(actor_params, obs_jax)[0]
        action = np.asarray(action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    print(f"  Reward: {episode_reward:.2f}")

env.close()