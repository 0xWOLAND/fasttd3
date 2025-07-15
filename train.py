import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import jax
import argparse
import pickle
import os

def save_model(agent, filepath):
    """Save the trained agent"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({
            'actor_params': agent.actor.params,
            'critic_params': agent.critic.params,
            'actor_target_params': agent.actor_target.params,
            'critic_target_params': agent.critic_target.params,
        }, f)
    print(f"Model saved to {filepath}")

def load_model(agent, filepath):
    """Load a trained agent"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    agent.actor = agent.actor.replace(params=checkpoint['actor_params'])
    agent.critic = agent.critic.replace(params=checkpoint['critic_params'])
    agent.actor_target = agent.actor_target.replace(params=checkpoint['actor_target_params'])
    agent.critic_target = agent.critic_target.replace(params=checkpoint['critic_target_params'])
    print(f"Model loaded from {filepath}")
    return agent

def eval_policy(agent, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed + 100)
        episode_reward = 0
        done = False
        while not done:
            action = np.asarray(agent.select_action(obs[None]))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        avg_reward += episode_reward
    
    avg_reward /= eval_episodes
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward

# Create vectorized environments - using humanoid task like FastTD3
num_envs = 128
env_name = "Humanoid-v5"  # MuJoCo humanoid task
print(f"Creating {num_envs} vectorized environments for {env_name}...")

try:
    envs = gym.make_vec(env_name, num_envs=num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    max_action = float(envs.single_action_space.high[0])
    print(f"Environment created successfully: obs_dim={obs_dim}, act_dim={act_dim}, max_action={max_action}")
except Exception as e:
    print(f"Error creating vectorized environment: {e}")
    raise

agent = TD3(
    state_dim=obs_dim,
    action_dim=act_dim,
    max_action=max_action,
    actor_def=Actor(obs_dim, act_dim, max_action, hidden_dim=256),
    critic_def=Critic(obs_dim, act_dim, num_atoms=51, hidden_dim=256, v_min=-10, v_max=10),
    num_envs=128,
)

rb = ReplayBuffer(obs_dim, act_dim, size=100_000, n_env=128, n_steps=1, gamma=0.99)

# Training parameters optimized for vectorized training
start_timesteps = 5000  # Reduced for faster learning with 128 envs
eval_freq = 5000  # How often (time steps) we evaluate
max_timesteps = 100000  # Reduced for faster testing
expl_noise = 0.1  # Std of Gaussian exploration noise
batch_size = 256  # Batch size for both actor and critic
seed = 0

# Evaluate untrained policy
evaluations = [eval_policy(agent, env_name, seed)]

obs, _ = envs.reset()  # Shape: [num_envs, obs_dim]
episode_rewards = np.zeros(num_envs)
episode_timesteps = np.zeros(num_envs)
episode_num = 0

print("Starting training loop...")

try:
    for t in range(max_timesteps):
        episode_timesteps += 1
        
        # Select actions for all environments
        if t < start_timesteps:
            actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        else:
            actions = np.asarray(agent.select_action(obs, add_noise=True))
        
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        dones = terminated | truncated
        episode_rewards += rewards
        
        # Handle episode termination vs truncation for proper done_bool  
        done_bools = terminated.astype(float)  # For Humanoid, use terminated for proper done signal
        
        rb.add(obs, actions, next_obs, rewards, done_bools)
        
        # Reset environments that are done
        if np.any(dones):
            finished_episodes = np.where(dones)[0]
            for env_idx in finished_episodes:
                print(f"Total T: {t+1} Env: {env_idx} Episode T: {episode_timesteps[env_idx]} Reward: {episode_rewards[env_idx]:.3f}")
                episode_rewards[env_idx] = 0
                episode_timesteps[env_idx] = 0
                episode_num += 1
        
        obs = next_obs

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            agent.train(rb, batch_size=batch_size)
        
        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            avg_reward = eval_policy(agent, env_name, seed)
            evaluations.append(avg_reward)
            
        # Progress update
        if t % 1000 == 0:
            print(f"Training step {t}/{max_timesteps}")

except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    raise

print(f"\nFinal evaluations: {evaluations}")
print(f"Average improvement: {evaluations[-1] - evaluations[0]:.3f}")

# Visualize trained agent
print("\nVisualizing trained agent...")
render_env = gym.make(env_name, render_mode="human")
obs, _ = render_env.reset()

for step in range(1000):  # Run for 1000 steps
    action = np.asarray(agent.select_action(obs[None], add_noise=False))[0]  # No noise for visualization
    obs, reward, terminated, truncated, _ = render_env.step(action)
    render_env.render()
    
    if terminated or truncated:
        obs, _ = render_env.reset()
        print(f"Episode finished at step {step}")

render_env.close()
print("Visualization complete!")
