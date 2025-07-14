import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import jax

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

# Create vectorized environments
num_envs = 128
envs = gym.make_vec("Pendulum-v1", num_envs=num_envs)
env = gym.make("Pendulum-v1")  # Single env for reference
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

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
evaluations = [eval_policy(agent, "Pendulum-v1", seed)]

obs, _ = envs.reset()  # Shape: [num_envs, obs_dim]
episode_rewards = np.zeros(num_envs)
episode_timesteps = np.zeros(num_envs)
episode_num = 0

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
    done_bools = terminated.astype(float)
    
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
        avg_reward = eval_policy(agent, "Pendulum-v1", seed)
        evaluations.append(avg_reward)

print(f"\nFinal evaluations: {evaluations}")
print(f"Average improvement: {evaluations[-1] - evaluations[0]:.3f}")
