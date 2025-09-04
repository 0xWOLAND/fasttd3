import mujoco_playground
from mujoco_playground import registry
import numpy as np
import jax
import jax.numpy as jnp
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import time
import pickle
import os
import uuid

# Helper to flatten observations
def flatten_obs(obs):
    if isinstance(obs, dict):
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1)
    return obs

def eval_policy(agent, env_name, seed, eval_episodes=3):
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

# Configuration
num_envs = 256
# env_name = "G1JoystickFlatTerrain"
env_name = "CheetahRun"
seed = 0
start_timesteps = 10
eval_freq = 5000
max_timesteps = 50_000
batch_size = 16384

# Setup environment
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)
dummy_state = env.reset(jax.random.PRNGKey(0))
obs_dim = sum(obs.shape[0] for obs in dummy_state.obs.values()) if isinstance(dummy_state.obs, dict) else env.observation_size
act_dim = env.action_size
max_action = 1.0

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step) 

# Create agent
agent = TD3(
    state_dim=obs_dim, action_dim=act_dim, max_action=max_action,
    actor_def=Actor(obs_dim, act_dim, max_action, hidden_dim=512),
    critic_def=Critic(obs_dim, act_dim, num_atoms=101, hidden_dim=1024, v_min=-10.0, v_max=10.0),
    num_envs=num_envs, num_updates=2,
    tau=0.1, policy_noise=0.001, noise_clip=0.5,
    actor_lr=3e-5, critic_lr=3e-5
)

rb = ReplayBuffer(obs_dim, act_dim, size=10240, n_env=num_envs, n_steps=1, gamma=0.97)

# Create run folder
run_id = f"{env_name}_{uuid.uuid4().hex[:8]}"
os.makedirs(run_id, exist_ok=True)

def save_checkpoint(agent, step, obs_dim, act_dim, max_action, episodes_completed, eval_reward):
    """Save checkpoint with eval reward"""
    filename = f"{run_id}/checkpoint_{step}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump({
            "actor_params": agent.actor.params,
            "obs_dim": obs_dim, "act_dim": act_dim, "max_action": max_action,
            "step": step, "episodes": episodes_completed, "eval_reward": eval_reward
        }, f)
    if eval_reward != 0.0:
        print(f"# Saved: {filename} (eval: {eval_reward:.2f})")
    else:
        print(f"# Saved: {filename}")

print("# FastTD3 Training")
print(f"# Config: envs={num_envs}, batch_size={batch_size}, eval_freq={eval_freq}")
print(f"# Columns: timestep,episode_reward,eval_reward,episodes_completed,wall_time,phase")

start_time = time.time()

# Initialize training
rng_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
states = jax.vmap(jit_reset)(rng_keys)
obs = jax.vmap(flatten_obs)(states.obs)
episode_rewards = np.zeros(num_envs)
episodes_completed = 0
recent_episode_rewards = []
for t in range(max_timesteps):
    # Select actions
    if t < start_timesteps:
        agent.rng, action_key = jax.random.split(agent.rng)
        actions = jax.random.uniform(action_key, (num_envs, act_dim), minval=-1.0, maxval=1.0)
    else:
        actions = agent.select_action(np.array(obs), add_noise=True)
    
    # Step all environments (vectorized)
    next_states = jax.vmap(jit_step)(states, actions)
    next_obs = jax.vmap(flatten_obs)(next_states.obs)
    rewards = next_states.reward
    dones = next_states.done
    episode_rewards += np.array(rewards)
    
    # Add to buffer
    rb.add(obs, actions, next_obs, rewards, dones.astype(float))
    
    # Reset finished environments (vectorized)
    if jnp.any(dones):
        agent.rng, reset_rng = jax.random.split(agent.rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        reset_states = jax.vmap(jit_reset)(reset_keys)
        
        def select_state(reset_val, next_val):
            mask_shape = (num_envs,) + (1,) * (len(next_val.shape) - 1)
            mask = dones.reshape(mask_shape)
            return jnp.where(mask, reset_val, next_val)
        
        next_states = jax.tree.map(select_state, reset_states, next_states)
        
        finished_episodes = jnp.where(dones)[0]
        episodes_completed += len(finished_episodes)
        recent_episode_rewards.extend(episode_rewards[finished_episodes].tolist())
        episode_rewards = jnp.where(dones, 0, episode_rewards)
    
    states = next_states
    obs = next_obs
    
    # Train agent
    if t >= start_timesteps and rb.size > max(batch_size * 2, 1000):
        agent.train(rb, batch_size=batch_size)
    
    # Logging & Evaluation
    if (t + 1) % 1000 == 0:
        avg_recent = np.mean(recent_episode_rewards[-10:]) if recent_episode_rewards else 0.0
        wall_time = time.time() - start_time
        phase = "T" if t >= start_timesteps else "E"
        
        # Run evaluation and checkpoint only when eval is done
        eval_reward = 0.0
        if (t + 1) % eval_freq == 0 and t >= start_timesteps:
            print(f"# Running evaluation at step {t+1}...")
            eval_reward = eval_policy(agent, env_name, seed, eval_episodes=3)
            print(f"# Evaluation: {eval_reward:.2f}")
            save_checkpoint(agent, t+1, obs_dim, act_dim, max_action, episodes_completed, eval_reward)
        
        print(f"{t+1},{avg_recent:.2f},{eval_reward:.2f},{episodes_completed},{wall_time:.2f},{phase}")

print(f"# Training complete! Episodes: {episodes_completed}")