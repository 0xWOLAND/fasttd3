import mujoco_playground
from mujoco_playground import registry
import numpy as np
import jax
import jax.numpy as jnp
from td3.td3 import TD3, Actor, Critic
from td3.utils import ReplayBuffer
import time

# Helper to flatten observations
def flatten_obs(obs):
    if isinstance(obs, dict):
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1)
    return obs

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

num_envs = 1
env_name = "G1JoystickFlatTerrain"
# env_name = "CheetahRun"
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

start_timesteps = 1000  
eval_freq = 5000  
max_timesteps = 5_000  
batch_size = 64
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

# Initialize training - single environment (no vectorization)
rng = jax.random.PRNGKey(seed)
state = jit_reset(rng)
obs = flatten_obs(state.obs)[None]  # Add batch dimension
states = [state]  # Keep as list for compatibility
episode_rewards = np.zeros(num_envs)
episodes_completed = 0
recent_episode_rewards = []
for t in range(max_timesteps):
    # Select actions
    if t < start_timesteps:
        agent.rng, action_key = jax.random.split(agent.rng)
        actions = jax.random.uniform(action_key, (num_envs, act_dim), minval=-max_action, maxval=max_action)
    else:
        actions = agent.select_action(np.array(obs), add_noise=True)
    
    # Step single environment
    next_state = jit_step(states[0], actions[0])
    next_obs = flatten_obs(next_state.obs)[None]  # Add batch dimension
    rewards = jnp.array([next_state.reward])
    dones = jnp.array([next_state.done])
    episode_rewards += np.array(rewards)
    
    # Add to buffer
    rb.add(obs, actions, next_obs, rewards, dones.astype(float))
    
    # Reset environment if done
    if dones[0]:
        agent.rng, reset_rng = jax.random.split(agent.rng)
        next_state = jit_reset(reset_rng)
        episodes_completed += 1
        recent_episode_rewards.append(episode_rewards[0])
        episode_rewards = np.array([0.0])
    
    states = [next_state]
    obs = next_obs
    
    # Train agent
    if t >= start_timesteps and rb.size > max(batch_size * 2, 1000):
        agent.train(rb, batch_size=batch_size)
    
    # Logging & Checkpointing
    if (t + 1) % 1000 == 0:
        avg_recent = np.mean(recent_episode_rewards[-10:]) if recent_episode_rewards else 0.0
        wall_time = time.time() - start_time
        phase = "T" if t >= start_timesteps else "E"
        print(f"{t+1},{avg_recent:.2f},0.0,{episodes_completed},{wall_time:.2f},{phase}")
        
        # Save checkpoint
        import pickle
        with open(f"checkpoint_{t+1}.pkl", "wb") as f:
            pickle.dump({
                "actor_params": agent.actor.params,
                "obs_dim": obs_dim, "act_dim": act_dim, "max_action": max_action,
                "step": t+1, "episodes": episodes_completed
            }, f)
    
    # Skip evaluation during training
    
# Training complete
print(f"# Training complete! Episodes: {episodes_completed}")

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