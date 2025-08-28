import numpy as np
import pickle
from td3.td3 import Actor
import jax
import jax.numpy as jnp
from brax import envs
from brax.io import html

# Load saved model
with open("trained_td3.pkl", "rb") as f:
    data = pickle.load(f)
    actor_params = data["actor_params"]
    obs_dim = data["obs_dim"]
    act_dim = data["act_dim"]
    max_action = data["max_action"]

actor = Actor(obs_dim, act_dim, max_action, hidden_dim=256)

# Create brax environment (same as training)
env = envs.create(
    env_name="humanoid",
    episode_length=1000,
    action_repeat=1,
    auto_reset=False,  # Don't auto-reset for visualization
    batch_size=1,
    backend="mjx"
)

# Run one episode and collect trajectory
rng = jax.random.PRNGKey(0)
state = env.reset(rng)
states = [state]
episode_reward = 0

print("Running episode for visualization...")

for step in range(1000):
    # Select action using trained policy
    action = actor.apply(actor_params, state.obs)
    action = jnp.clip(action, -max_action, max_action)
    
    state = env.step(state, action)
    states.append(state)
    episode_reward += float(state.reward[0])
    
    if state.done[0]:
        break

print(f"Episode reward: {episode_reward:.2f}")
print(f"Episode length: {len(states)} steps")

# Create HTML visualization
print("Generating visualization...")
html_string = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), states)

# Save to file
with open("policy_visualization.html", "w") as f:
    f.write(html_string)

print("Visualization saved to policy_visualization.html")
print("Open this file in a web browser to view the simulation!")