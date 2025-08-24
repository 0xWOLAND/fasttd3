import gymnasium as gym
import numpy as np
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import jax
import time

def eval_policy(agent, env_name, seed, eval_episodes=10):
    """Evaluate the policy for a given number of episodes."""
    # Use vectorized environments for faster evaluation
    if eval_episodes > 1:
        eval_envs = gym.make_vec(env_name, num_envs=min(eval_episodes, 4))
        num_envs = eval_envs.num_envs
        episodes_per_env = (eval_episodes + num_envs - 1) // num_envs
        
        total_rewards = []
        obs, _ = eval_envs.reset(seed=seed)
        episode_rewards = np.zeros(num_envs)
        episodes_completed = np.zeros(num_envs, dtype=int)
        
        while np.any(episodes_completed < episodes_per_env):
            # Get actions for all environments at once
            actions = np.asarray(agent.select_action(obs))
            obs, rewards, terminated, truncated, _ = eval_envs.step(actions)
            dones = terminated | truncated
            
            # Update rewards
            episode_rewards += rewards
            
            # Handle completed episodes
            if np.any(dones):
                for idx in np.where(dones)[0]:
                    if episodes_completed[idx] < episodes_per_env:
                        total_rewards.append(episode_rewards[idx])
                        episode_rewards[idx] = 0
                        episodes_completed[idx] += 1
        
        return np.mean(total_rewards[:eval_episodes])
    else:
        # Single episode evaluation
        eval_env = gym.make(env_name)
        obs, _ = eval_env.reset(seed=seed)
        episode_reward = 0
        done = False
        while not done:
            action = np.asarray(agent.select_action(obs[None]))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        return episode_reward

# Create environments
num_envs = 8 
envs = gym.make_vec("HalfCheetah-v5", num_envs=num_envs)
env = gym.make("HalfCheetah-v5")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

n_steps = 3  
num_updates = 2 

agent = TD3(
    state_dim=obs_dim,
    action_dim=act_dim,
    max_action=max_action,
    actor_def=Actor(obs_dim, act_dim, max_action, hidden_dim=256),
    critic_def=Critic(obs_dim, act_dim, num_atoms=51, hidden_dim=256, v_min=-2000, v_max=2000),
    num_envs=num_envs,
    num_updates=num_updates,
    tau=0.005,
    policy_noise=0.2,  # Standard TD3 noise
    noise_clip=0.5,    # Standard TD3 clip
    actor_lr=3e-4,
    critic_lr=3e-4,
)

rb = ReplayBuffer(obs_dim, act_dim, size=10_000, n_env=num_envs, n_steps=n_steps, gamma=0.99)

start_timesteps = 2000  
eval_freq = 5000  
max_timesteps = 20_000  
batch_size = 256
seed = 0

print("# FastTD3 Training")
print(f"# Config: envs={num_envs}, n_steps={n_steps}, num_updates={num_updates}, batch_size={batch_size}")
print(f"# Episode length: 1000 steps (HalfCheetah-v5)")
print(f"# Columns: timestep,episode_reward,eval_reward,episodes_completed,wall_time,phase")
print("# Data:")

# Track metrics
start_time = time.time()
episode_rewards_tracker = []
eval_rewards_tracker = []

# Evaluate untrained policy
initial_eval = eval_policy(agent, "HalfCheetah-v5", seed, eval_episodes=3)
print(f"0,0.0,{initial_eval:.2f},0,0.00")

# Initialize training
obs, _ = envs.reset()
episode_rewards = np.zeros(num_envs)
episode_timesteps = np.zeros(num_envs)
episodes_completed = 0
recent_episode_rewards = []

actions = np.zeros((num_envs, act_dim), dtype=np.float32)

for t in range(max_timesteps):
    episode_timesteps += 1
    
    # Select actions for all environments
    if t < start_timesteps:
        # Random exploration (vectorized)
        actions[:] = jax.random.uniform(agent.rng, (num_envs, act_dim), minval=-max_action, maxval=max_action)
        agent.rng, _ = jax.random.split(agent.rng)
    else:
        # Policy with exploration noise (already vectorized)
        actions[:] = agent.select_action(obs, add_noise=True)
    
    # Step all environments
    next_obs, rewards, terminated, truncated, _ = envs.step(actions)
    dones = terminated | truncated
    episode_rewards += rewards
    
    # Handle episode termination vs truncation for proper done_bool
    done_bools = terminated.astype(float)
    
    rb.add(obs, actions, next_obs, rewards, done_bools)
    
    if np.any(dones):
        finished_episodes = np.where(dones)[0]
        episodes_completed += len(finished_episodes)
        recent_episode_rewards.extend(episode_rewards[finished_episodes].tolist())
        episode_rewards[dones] = 0
        episode_timesteps[dones] = 0
    
    obs = next_obs
    
    # Train agent after collecting sufficient data
    min_buffer_size = max(batch_size * 2, 1000)  
    if t >= start_timesteps and rb.size > min_buffer_size:
        agent.train(rb, batch_size=batch_size)
    
    if (t + 1) % 1000 == 0:
        if recent_episode_rewards:
            avg_recent = np.mean(recent_episode_rewards[-10:])  # Last 10 episodes
        else:
            avg_recent = 0.0
        
        wall_time = time.time() - start_time
        is_training = "T" if t >= start_timesteps else "E"  # T=training, E=exploring
        print(f"{t+1},{avg_recent:.2f},0.0,{episodes_completed},{wall_time:.2f},{is_training}")
    
    if (t + 1) % eval_freq == 0:
        eval_reward = eval_policy(agent, "HalfCheetah-v5", seed, eval_episodes=3)
        wall_time = time.time() - start_time
        
        if recent_episode_rewards:
            avg_recent = np.mean(recent_episode_rewards[-10:])
        else:
            avg_recent = 0.0
        
        print(f"{t+1},{avg_recent:.2f},{eval_reward:.2f},{episodes_completed},{wall_time:.2f}")

final_eval = eval_policy(agent, "HalfCheetah-v5", seed, eval_episodes=5)
wall_time = time.time() - start_time
if recent_episode_rewards:
    avg_recent = np.mean(recent_episode_rewards[-10:])
else:
    avg_recent = 0.0

print(f"{max_timesteps},{avg_recent:.2f},{final_eval:.2f},{episodes_completed},{wall_time:.2f}")
print(f"# Training complete! Final eval: {final_eval:.2f}, Episodes: {episodes_completed}")

# Save the trained model
import pickle
with open("trained_td3.pkl", "wb") as f:
    pickle.dump({
        "actor_params": agent.actor.params,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "max_action": max_action
    }, f)
print("# Model saved to trained_td3.pkl")