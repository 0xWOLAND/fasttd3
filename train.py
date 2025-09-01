import mujoco_playground
from mujoco_playground import registry
import numpy as np
import jax
import jax.numpy as jnp
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import time

def eval_policy(agent, env_name, seed, eval_episodes=10):
    """Evaluate the policy for a given number of episodes."""
    eval_env_cfg = registry.get_default_config(env_name)
    eval_env = registry.load(env_name, config=eval_env_cfg)
    jit_eval_reset = jax.jit(eval_env.reset)
    jit_eval_step = jax.jit(eval_env.step)
    
    total_reward = 0
    for i in range(eval_episodes):
        rng = jax.random.PRNGKey(seed + i)
        state = jit_eval_reset(rng)
        episode_reward = 0
        while not state.done:
            flat_obs = flatten_obs(state.obs)
            action = agent.select_action(np.array(flat_obs)[None])[0]
            state = jit_eval_step(state, action)
            episode_reward += state.reward
        total_reward += episode_reward
    return total_reward / eval_episodes

# Create environment
num_envs = 8
env_name = "G1JoystickFlatTerrain"
env_cfg = registry.get_default_config(env_name) 
env = registry.load(env_name, config=env_cfg)
# Flatten observation space for nested obs
dummy_state = env.reset(jax.random.PRNGKey(0))
if isinstance(dummy_state.obs, dict):
    obs_dim = sum(obs.shape[0] for obs in dummy_state.obs.values())
else:
    obs_dim = env.observation_size
act_dim = env.action_size
max_action = 1.0

# JIT compile environment functions for performance
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

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
print(f"# Episode length: 1000 steps ({env_name})")
print(f"# Columns: timestep,episode_reward,eval_reward,episodes_completed,wall_time,phase")
print("# Data:")

# Track metrics
start_time = time.time()
episode_rewards_tracker = []
eval_rewards_tracker = []

# Evaluate untrained policy
initial_eval = eval_policy(agent, env_name, seed, eval_episodes=3)
print(f"0,0.0,{initial_eval:.2f},0,0.00")

# Helper to flatten observations
def flatten_obs(obs):
    if isinstance(obs, dict):
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1)
    return obs

# Initialize training - vectorized environments
rng_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
states = jax.vmap(jit_reset)(rng_keys)
obs = jax.vmap(flatten_obs)(states.obs)
episode_rewards = np.zeros(num_envs)
episode_timesteps = np.zeros(num_envs) 
episodes_completed = 0
recent_episode_rewards = []

for t in range(max_timesteps):
    episode_timesteps += 1
    
    # Select actions for all environments
    if t < start_timesteps:
        # Random exploration
        agent.rng, action_key = jax.random.split(agent.rng)
        actions = jax.random.uniform(action_key, (num_envs, act_dim), minval=-max_action, maxval=max_action)
    else:
        # Policy with exploration noise
        actions = agent.select_action(np.array(obs), add_noise=True)
    
    # Step all environments (vectorized)
    next_states = jax.vmap(jit_step)(states, actions)
    next_obs = jax.vmap(flatten_obs)(next_states.obs)
    rewards = next_states.reward
    dones = next_states.done
    
    episode_rewards += np.array(rewards)
    
    # Add to replay buffer
    rb.add(obs, actions, next_obs, rewards, dones.astype(float))
    
    # Reset finished environments (vectorized)
    num_resets = jnp.sum(dones)
    if num_resets > 0:
        agent.rng, reset_rng = jax.random.split(agent.rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        reset_states = jax.vmap(jit_reset)(reset_keys)
        next_states = jax.tree_map(
            lambda new, old, mask: jnp.where(mask[..., None], new, old),
            reset_states, next_states, dones
        )
    
    states = next_states
    obs = next_obs
    
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
        eval_reward = eval_policy(agent, env_name, seed, eval_episodes=3)
        wall_time = time.time() - start_time
        
        if recent_episode_rewards:
            avg_recent = np.mean(recent_episode_rewards[-10:])
        else:
            avg_recent = 0.0
        
        print(f"{t+1},{avg_recent:.2f},{eval_reward:.2f},{episodes_completed},{wall_time:.2f}")

final_eval = eval_policy(agent, env_name, seed, eval_episodes=5)
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