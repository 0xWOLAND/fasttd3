import os
# Prevent JAX from allocating too much GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use only 50% of GPU memory

import gymnasium as gym
import numpy as np
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
from td3.normalization import EmpiricalNormalization, RewardNormalizer

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
    
    eval_env.close() 
    avg_reward = np.mean(rewards)
    print(f"Evaluation: {avg_reward:.1f}")
    return avg_reward

num_envs, max_timesteps, eval_freq = 64, 150000, 5000
learning_starts, batch_size, buffer_size = 10, 8192, 16384 
gamma, lr, expl_noise = 0.99, 3e-4, 0.001  
num_updates = 2  
tau = 0.1  
policy_frequency = 2  

# Setup
envs = gym.make_vec("Humanoid-v5", num_envs=num_envs)
obs_dim, act_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
max_action = float(envs.single_action_space.high[0])

# Create normalizers
obs_normalizer = EmpiricalNormalization(obs_dim)
reward_normalizer = RewardNormalizer()


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

noise_scales = np.random.uniform(0.001, 0.4, (num_envs, 1))

for t in range(max_timesteps):
    if t < learning_starts:
        actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
    else:
        actions = np.asarray(agent.select_action(obs, add_noise=False))
        noise = np.random.randn(*actions.shape) * noise_scales * max_action
        actions = np.clip(actions + noise, -max_action, max_action)
    
    next_obs, rewards, terminated, truncated, _ = envs.step(actions)
    dones = terminated | truncated
    episode_rewards += rewards
    
    transition = {
        "observations": obs,
        "actions": actions,
        "critic_observations": obs, 
        "next": {
            "observations": next_obs,
            "rewards": rewards,
            "dones": terminated.astype(int),
            "truncations": truncated.astype(int),
            "critic_observations": next_obs
        }
    }
    rb.add(transition)
    
    if np.any(dones):
        noise_scales[dones] = np.random.uniform(0.001, 0.4, (np.sum(dones), 1))
        
        finished_rewards = episode_rewards[dones]
        recent_rewards.extend(finished_rewards)
        recent_rewards = recent_rewards[-200:] 
        
        reward_normalizer.update(finished_rewards)
        
        if len(recent_rewards) >= 10 and len(recent_rewards) % 20 == 0:  
            avg_recent = np.mean(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.mean(recent_rewards)
            best_recent = np.max(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.max(recent_rewards)
            worst_recent = np.min(recent_rewards[-50:]) if len(recent_rewards) >= 50 else np.min(recent_rewards)
            print(f"T: {t+1:5d} | Episodes: {len(recent_rewards):3d} | Avg(50): {avg_recent:7.1f} | Best(50): {best_recent:7.1f} | Worst(50): {worst_recent:7.1f}")
        
        episode_rewards[dones] = 0
    
    obs_normalizer.update(obs)
    
    obs = next_obs
    if t >= learning_starts: 
        for update_i in range(num_updates):
            batch = rb.sample(agent.rng, batch_size)
            norm_obs = obs_normalizer.normalize(batch["observations"])
            norm_next_obs = obs_normalizer.normalize(batch["next"]["observations"])
            
            norm_rewards = reward_normalizer.normalize(batch["next"]["rewards"])
            
            from td3.utils import Transition
            normalized_batch = Transition(
                norm_obs, batch["actions"], norm_next_obs, 
                norm_rewards, batch["next"]["dones"], 
                batch["next"]["effective_n_steps"]
            )
            
            agent.train_batch(normalized_batch)
            
        if t % 1000 == 0:
            print(f"Debug: Step {t}, total_it={agent.total_it}")
    if (t + 1) % eval_freq == 0: 
        evaluations.append(eval_policy(agent, 0))
        import pickle
        with open(f"checkpoint_{t+1}.pkl", "wb") as f:
            pickle.dump({"actor_params": agent.actor.params, "critic_params": agent.critic.params}, f)

print(f"Final: {evaluations}")
print(f"Improvement: {evaluations[-1] - evaluations[0]:.3f}")
