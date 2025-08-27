print("Starting imports...")
from brax import envs
import numpy as np
print("Basic imports done...")
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import jax
import jax.numpy as jnp
import time
from jax import profiler
print("All imports complete!")


def eval_policy(agent, env, seed, eval_episodes=3):
    num_parallel = env.batch_size  
    
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    state = env.reset(reset_key)
    episode_rewards = jnp.zeros(num_parallel)
    completed_episodes = []
    
    while len(completed_episodes) < eval_episodes:
        actions = agent.select_action(state.obs, add_noise=False)
        state = env.step(state, actions)
        episode_rewards += state.reward
        
        if state.done.any():
            done_indices = jnp.where(state.done)[0]
            for idx in done_indices:
                if len(completed_episodes) < eval_episodes:
                    completed_episodes.append(float(episode_rewards[idx]))
            episode_rewards = jnp.where(state.done, 0.0, episode_rewards)
    
    return float(jnp.mean(jnp.array(completed_episodes)))

num_envs = 2  # Reduce to test if memory/batch size is the issue
env = envs.create(
    env_name="humanoid",
    episode_length=1000,
    action_repeat=1,
    auto_reset=True,
    batch_size=num_envs,
    backend="mjx"
)

obs_dim = env.observation_size
act_dim = env.action_size
max_action = 1.0

n_steps = 3  
num_updates = 2 

agent = TD3(
    state_dim=obs_dim,
    action_dim=act_dim,
    max_action=max_action,
    actor_def=Actor(obs_dim, act_dim, max_action, hidden_dim=256),
    critic_def=Critic(obs_dim, act_dim, num_atoms=32, hidden_dim=256, v_min=-1000, v_max=7000),
    num_envs=num_envs,
    num_updates=1,  # Reduce updates
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    actor_lr=3e-4,
    critic_lr=3e-4,
)

rb = ReplayBuffer(obs_dim, act_dim, size=100_000, n_env=num_envs, n_steps=n_steps, gamma=0.99)

start_timesteps = 25_000  
eval_freq = 5000  
max_timesteps = 1_000_000  
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
print("# Evaluating initial policy...")
initial_eval = eval_policy(agent, env, seed, eval_episodes=3)
print(f"0,0.0,{initial_eval:.2f},0,0.00")

# Initialize training
print("# Resetting environments...")
rng = jax.random.PRNGKey(seed)
rng, reset_key = jax.random.split(rng)
state = env.reset(reset_key)
print("# Starting training loop...")

for t in range(max_timesteps):
    print(f'{t}/{max_timesteps}')
    # Progress indicator for exploration phase
    if t == 0:
        print(f"# Exploration phase: 0/{start_timesteps} steps")
    elif t < start_timesteps and (t + 1) % 5000 == 0:
        print(f"# Exploration phase: {t+1}/{start_timesteps} steps")
    elif t == start_timesteps:
        print(f"# Training phase started at step {t+1}")
    
    # Select actions for all environments
    if t < start_timesteps:
        # Random exploration
        rng, action_key = jax.random.split(rng)
        actions = jax.random.uniform(action_key, (num_envs, act_dim), minval=-max_action, maxval=max_action)
    else:
        # Policy with exploration noise
        actions = agent.select_action(state.obs, add_noise=True)
    
    # Step environments (auto-reset handles done environments)
    state = env.step(state, actions)
    
    # Add to replay buffer - keep on GPU (no CPU conversion)
    rb.add(
        state.obs, 
        actions, 
        state.obs,  # With auto-reset, obs is already "next_obs"
        state.reward, 
        state.done.astype(float)
    )
    
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
        eval_reward = eval_policy(agent, env, seed, eval_episodes=3)
        wall_time = time.time() - start_time
        
        if recent_episode_rewards:
            avg_recent = np.mean(recent_episode_rewards[-10:])
        else:
            avg_recent = 0.0
        
        print(f"{t+1},{avg_recent:.2f},{eval_reward:.2f},{episodes_completed},{wall_time:.2f}")

final_eval = eval_policy(agent, env, seed, eval_episodes=5)
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