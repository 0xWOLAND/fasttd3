from typing import NamedTuple
import jax
import jax.numpy as jnp

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    effective_n: jnp.ndarray


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, n_env, n_steps, gamma):
        self.obs = jnp.zeros((n_env, size, obs_dim))
        self.next_obs = jnp.zeros_like(self.obs)
        self.action = jnp.zeros((n_env, size, act_dim))
        self.reward = jnp.zeros((n_env, size))
        self.done = jnp.zeros((n_env, size))
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.n_env = n_env
        self.n_steps = n_steps
        self.gamma = gamma

    def add(self, obs, action, next_obs, reward, done):
        self.obs = self.obs.at[:, self.ptr].set(obs)
        self.action = self.action.at[:, self.ptr].set(action)
        self.next_obs = self.next_obs.at[:, self.ptr].set(next_obs)
        self.reward = self.reward.at[:, self.ptr].set(reward)
        self.done = self.done.at[:, self.ptr].set(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: jax.random.PRNGKey, batch_size: int) -> Transition:
        # Sample from flattened buffer like FastTD3
        total_samples = self.n_env * self.size
        flat_idx = jax.random.randint(rng, (batch_size,), 0, total_samples)
        env_idx = flat_idx // self.size
        buffer_idx = flat_idx % self.size

        obs = self.obs[env_idx, buffer_idx]
        act = self.action[env_idx, buffer_idx]
        next_obs = self.next_obs[env_idx, buffer_idx]
        reward = self.reward[env_idx, buffer_idx]
        done = self.done[env_idx, buffer_idx]

        return Transition(obs, act, next_obs, reward, done, jnp.ones_like(reward))
