import gymnasium as gym
import numpy as np
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer

def eval_policy(agent, seed, eval_episodes=5):
    eval_env = gym.make("Humanoid-v4")
    rewards = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed + 100)
        episode_reward, done = 0, False
        while not done:
            action = np.asarray(agent.select_action(obs[None]))[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    avg_reward = np.mean(rewards)
    print(f"Evaluation: {avg_reward:.1f}")
    return avg_reward

num_envs, max_timesteps, eval_freq = 32, 50000, 5000
learning_starts, batch_size, buffer_size = 10, 4096, 8192
gamma, lr, expl_noise = 0.99, 3e-4, 0.4

# Setup
envs = gym.make_vec("Humanoid-v4", num_envs=num_envs)
obs_dim, act_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
max_action = float(envs.single_action_space.high[0])

agent = TD3(obs_dim, act_dim, max_action, 
           Actor(obs_dim, act_dim, max_action, hidden_dim=512),
           Critic(obs_dim, act_dim, num_atoms=101, hidden_dim=1024, v_min=-250, v_max=250), num_envs)

rb = ReplayBuffer(obs_dim, act_dim, buffer_size, num_envs, 1, gamma)
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
    
    rb.add(obs, actions, next_obs, rewards, terminated.astype(float))
    
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
    
    obs = next_obs
    if t >= learning_starts: 
        agent.train(rb, batch_size)
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
