import gymnasium as gym
import numpy as np
import pickle
import sys
from td3.td3 import TD3, Actor, Critic

# Load checkpoint
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_5000.pkl"
with open(checkpoint_path, "rb") as f:
    checkpoint = pickle.load(f)

# Create agent
env = gym.make("Humanoid-v4", render_mode="human")
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3(obs_dim, act_dim, max_action, 
           Actor(obs_dim, act_dim, max_action, hidden_dim=512),
           Critic(obs_dim, act_dim, num_atoms=101, hidden_dim=1024, v_min=-250, v_max=250), 1)

# Load parameters
agent.actor = agent.actor.replace(params=checkpoint["actor_params"])

# Visualize
for _ in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.asarray(agent.select_action(obs[None]))[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode reward: {total_reward:.1f}")

env.close()