import mujoco_playground
from mujoco_playground import registry
import numpy as np
import pickle
from td3.td3 import Actor
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import time

# Helper to flatten observations
def flatten_obs(obs):
    if isinstance(obs, dict):
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1)
    return obs

# Load saved model
with open("checkpoint_4000.pkl", "rb") as f:
    data = pickle.load(f)
    actor_params = data["actor_params"]
    obs_dim = data["obs_dim"]
    act_dim = data["act_dim"]
    max_action = data["max_action"]

actor = Actor(obs_dim, act_dim, max_action, hidden_dim=256)
env_name = "CheetahRun"
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)

print(f"Running trained policy on {env_name} with visualization...")

# Get the MuJoCo model and create viewer once
setup_start = time.time()
model = env.mj_model
data = mujoco.MjData(model)
setup_time = time.time() - setup_start
print(f"Setup time: {setup_time:.4f}s")

print("Launching MuJoCo viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Viewer running: {viewer.is_running()}")
    for episode in range(3):
        episode_start = time.time()
        print(f"Episode {episode + 1}")
        
        reset_start = time.time()
        rng = jax.random.PRNGKey(episode)
        state = jit_reset(rng)
        reset_time = time.time() - reset_start
        
        episode_reward = 0
        
        # Initialize MuJoCo data with the initial state
        init_start = time.time()
        data.qpos[:] = state.data.qpos
        data.qvel[:] = state.data.qvel
        data.time = 0
        init_time = time.time() - init_start
        
        step_count = 0
        policy_time = 0
        env_step_time = 0
        viewer_time = 0
        
        while not state.done and viewer.is_running():
            # Policy inference timing
            policy_start = time.time()
            flat_obs = flatten_obs(state.obs)
            obs_jax = jnp.array(flat_obs[None])
            action = actor.apply(actor_params, obs_jax)[0]
            action = np.asarray(action)
            policy_time += time.time() - policy_start
            
            # Environment step timing
            env_start = time.time()
            state = jit_step(state, action)
            episode_reward += state.reward
            env_step_time += time.time() - env_start
            
            # Viewer update timing - run MuJoCo physics too
            viewer_start = time.time()
            # Set control inputs
            data.ctrl[:] = action
            # Copy state from environment
            data.qpos[:] = state.data.qpos
            data.qvel[:] = state.data.qvel
            data.time = step_count * env.dt
            # Step MuJoCo physics for visualization
            mujoco.mj_step(model, data)
            viewer.sync()
            viewer_time += time.time() - viewer_start
            
            step_count += 1
            
            # Debug: Print position changes every 1000 steps
            if step_count % 1000 == 0:
                print(f"    qpos[0:3]: {data.qpos[:3]}")  # First 3 positions
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {episode_reward:.2f}")
        
        episode_time = time.time() - episode_start
        print(f"  Final Reward: {episode_reward:.2f}")
        print(f"  Episode timing:")
        print(f"    Reset: {reset_time:.4f}s")
        print(f"    Init: {init_time:.4f}s")
        print(f"    Policy inference: {policy_time:.4f}s ({policy_time/step_count:.6f}s per step)")
        print(f"    Environment steps: {env_step_time:.4f}s ({env_step_time/step_count:.6f}s per step)")
        print(f"    Viewer updates: {viewer_time:.4f}s ({viewer_time/step_count:.6f}s per step)")
        print(f"    Total episode: {episode_time:.4f}s")
        print(f"    Steps per second: {step_count/episode_time:.2f}")
        time.sleep(1)  # Brief pause between episodes