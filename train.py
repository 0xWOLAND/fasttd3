import os
# Prevent JAX from allocating too much GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use only 50% of GPU memory

import gymnasium as gym
import numpy as np
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer

def eval_policy(agent, seed, eval_episodes=5):
    eval_env = gym.make("Humanoid-v5")
    rewards = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed + 100)
        episode_reward, done = 0, False
        step_count = 0
        max_steps = 1000  # Prevent infinite episodes
        
        while not done and step_count < max_steps:
            action = np.asarray(agent.select_action(obs[None]))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
            
        rewards.append(episode_reward)
    
    eval_env.close()  # Clean up environment
    avg_reward = np.mean(rewards)
    print(f"Evaluation: {avg_reward:.1f}")
    return avg_reward

num_envs, max_timesteps, eval_freq = 64, 150000, 5000
learning_starts, batch_size, buffer_size = 1000, 8192, 16384
gamma, lr, expl_noise = 0.99, 1e-4, 0.001  # Back to simple setup
num_updates = 2  # FastTD3 uses 2 updates per step
tau = 0.1  # FastTD3 aggressive target updates
policy_frequency = 2  # Policy update frequency

# Setup
envs = gym.make_vec("Humanoid-v5", num_envs=num_envs)
obs_dim, act_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
max_action = float(envs.single_action_space.high[0])

# Observation normalization (running statistics)
obs_mean, obs_std = np.zeros(obs_dim), np.ones(obs_dim)
obs_count = 1e-4

# Adaptive exploration noise per environment
noise_scales = np.random.uniform(0.05, 0.8, (num_envs, 1))  # Random noise scale per env

# Reward normalization
reward_running_mean = 0.0
reward_running_var = 1.0
reward_count = 1e-4

def normalize_obs(obs):
    return (obs - obs_mean) / (obs_std + 1e-8)

def update_obs_stats(obs):
    global obs_mean, obs_std, obs_count
    batch_mean = np.mean(obs, axis=0)
    batch_var = np.var(obs, axis=0)
    batch_count = obs.shape[0]
    
    # More conservative updates to prevent drift - freeze after sufficient data
    if obs_count < 50000:  # Only update for first 50k samples
        update_rate = min(0.005, batch_count / (obs_count + batch_count))  # Max 0.5% update
        obs_mean = (1 - update_rate) * obs_mean + update_rate * batch_mean
        obs_std = (1 - update_rate) * obs_std + update_rate * np.sqrt(batch_var)
    
    # Prevent std from becoming too small or too large
    obs_std = np.clip(obs_std, 0.01, 10.0)
    obs_count += batch_count

def update_reward_stats(rewards):
    global reward_running_mean, reward_running_var, reward_count
    batch_count = len(rewards)
    batch_mean = np.mean(rewards)
    batch_var = np.var(rewards)
    
    # Update running statistics
    update_rate = min(0.01, batch_count / (reward_count + batch_count))
    reward_running_mean = (1 - update_rate) * reward_running_mean + update_rate * batch_mean
    reward_running_var = (1 - update_rate) * reward_running_var + update_rate * batch_var
    reward_count += batch_count

def normalize_reward(reward):
    return (reward - reward_running_mean) / (np.sqrt(reward_running_var) + 1e-8)

def get_cosine_lr(step, total_steps, initial_lr, final_lr):
    """Cosine annealing learning rate schedule"""
    progress = step / total_steps
    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
    return final_lr + (initial_lr - final_lr) * cosine_factor

agent = TD3(obs_dim, act_dim, max_action, 
           Actor(obs_dim, act_dim, max_action, hidden_dim=512),
           Critic(obs_dim, act_dim, num_atoms=101, hidden_dim=1024, v_min=-250, v_max=250), 
           num_envs, expl_noise=expl_noise, tau=tau, policy_freq=policy_frequency, 
           actor_lr=lr, critic_lr=lr)

rb = ReplayBuffer(n_env=num_envs, buffer_size=buffer_size, n_obs=obs_dim, n_act=act_dim, 
                  n_steps=1, gamma=gamma)
evaluations = [eval_policy(agent, 0)]

obs, _ = envs.reset()
episode_rewards = np.zeros(num_envs)
recent_rewards = []

for t in range(max_timesteps):
    actions = (np.array([envs.single_action_space.sample() for _ in range(num_envs)]) 
              if t < learning_starts else np.asarray(agent.select_action(obs, add_noise=True)))
    
    next_obs, rewards, terminated, truncated, _ = envs.step(actions)
    dones = terminated | truncated
    episode_rewards += rewards
    
    # Create transition dictionary for replay buffer
    transition = {
        "observations": obs,
        "actions": actions,
        "critic_observations": obs,  # Same as obs since no asymmetric observations
        "next": {
            "observations": next_obs,
            "rewards": rewards,
            "dones": terminated.astype(int),
            "truncations": truncated.astype(int),
            "critic_observations": next_obs
        }
    }
    rb.add(transition)
    
    # Track completed episodes with running statistics
    if np.any(dones):
        finished_rewards = episode_rewards[dones]
        recent_rewards.extend(finished_rewards)
        recent_rewards = recent_rewards[-200:]  # Keep last 200 episodes
        
        if len(recent_rewards) >= 10 and len(recent_rewards) % 20 == 0:  # Print every 20 episodes
            avg_recent = np.mean(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.mean(recent_rewards)
            best_recent = np.max(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.max(recent_rewards)
            worst_recent = np.min(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.min(recent_rewards)
            print(f"T: {t+1:5d} | Episodes: {len(recent_rewards):3d} | Avg(50): {avg_recent:7.1f} | Best(50): {best_recent:7.1f} | Worst(50): {worst_recent:7.1f}")
        
        episode_rewards[dones] = 0
    
    # Update observation normalization stats
    update_obs_stats(obs)
    
    obs = next_obs
    if t >= learning_starts: 
        # Multiple updates per step (FastTD3 approach)
        for update_i in range(num_updates):
            # Sample and normalize batch
            batch = rb.sample(agent.rng, batch_size)
            # Normalize observations
            norm_obs = normalize_obs(batch["observations"])
            norm_next_obs = normalize_obs(batch["next"]["observations"])
            
            # Update batch with normalized observations
            from td3.utils import Transition
            normalized_batch = Transition(
                norm_obs, batch["actions"], norm_next_obs, 
                batch["next"]["rewards"], batch["next"]["dones"], 
                batch["next"]["effective_n_steps"]
            )
            
            agent.train_batch(normalized_batch)
            
        # Debug: print loss occasionally
        if t % 1000 == 0:
            print(f"Debug: Step {t}, total_it={agent.total_it}")
    if (t + 1) % eval_freq == 0: 
        evaluations.append(eval_policy(agent, 0))
        # Save checkpoint
        import pickle
        with open(f"checkpoint_{t+1}.pkl", "wb") as f:
            pickle.dump({"actor_params": agent.actor.params, "critic_params": agent.critic.params}, f)

print(f"Final: {evaluations}")
print(f"Improvement: {evaluations[-1] - evaluations[0]:.3f}")
