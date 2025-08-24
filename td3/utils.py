from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    effective_n: jnp.ndarray


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, n_env, n_steps, gamma):
        # Use numpy arrays for storage to avoid JAX compilation overhead
        self.obs = np.zeros((n_env, size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((n_env, size, obs_dim), dtype=np.float32)
        self.action = np.zeros((n_env, size, act_dim), dtype=np.float32)
        self.reward = np.zeros((n_env, size), dtype=np.float32)
        self.done = np.zeros((n_env, size), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.n_env = n_env
        self.n_steps = n_steps
        self.gamma = gamma

    def add(self, obs, action, next_obs, reward, done):
        # Convert to numpy if needed and store
        self.obs[:, self.ptr] = np.asarray(obs)
        self.action[:, self.ptr] = np.asarray(action)
        self.next_obs[:, self.ptr] = np.asarray(next_obs)
        self.reward[:, self.ptr] = np.asarray(reward)
        self.done[:, self.ptr] = np.asarray(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: jax.random.PRNGKey, batch_size: int) -> Transition:
        # Ensure we have enough samples
        if self.size <= self.n_steps:
            raise ValueError(f"Not enough samples in buffer: {self.size} <= {self.n_steps}")
        
        # Convert numpy arrays to JAX arrays for sampling
        obs_jax = jnp.asarray(self.obs)
        next_obs_jax = jnp.asarray(self.next_obs)
        action_jax = jnp.asarray(self.action)
        reward_jax = jnp.asarray(self.reward)
        done_jax = jnp.asarray(self.done)
        
        # Sample batch_size total transitions across all environments
        batch_per_env = max(1, batch_size // self.n_env)
        idx = jax.random.randint(rng, (self.n_env, batch_per_env), 0, self.size - self.n_steps)
        steps = jnp.arange(self.n_steps)[None, None, :]
        all_idx = (idx[..., None] + steps) % self.max_size  # [env, B, n_step]

        rewards = jnp.take_along_axis(reward_jax[..., None], all_idx, axis=1)
        dones = jnp.take_along_axis(done_jax[..., None], all_idx, axis=1)

        mask = jnp.cumprod(1.0 - jnp.pad(dones[..., :-1], ((0, 0), (0, 0), (1, 0))), axis=-1)
        discounted = rewards * (mask * self.gamma ** steps)
        returns = discounted.sum(-1)

        done_anywhere = dones.any(-1)
        first_done_idx = jnp.argmax(dones, axis=-1)
        fallback_idx = all_idx[..., -1]
        flat_env = jnp.arange(self.n_env)[:, None]
        flat_batch = jnp.arange(batch_per_env)[None, :]
        next_idx = jnp.where(
            done_anywhere,
            all_idx[flat_env, flat_batch, first_done_idx],
            fallback_idx,
        )

        obs = jnp.take_along_axis(obs_jax, idx[..., None], axis=1).reshape(-1, obs_jax.shape[-1])
        act = jnp.take_along_axis(action_jax, idx[..., None], axis=1).reshape(-1, action_jax.shape[-1])
        next_obs = jnp.take_along_axis(next_obs_jax, next_idx[..., None], axis=1).reshape(-1, obs_jax.shape[-1])
        dones = jnp.take_along_axis(done_jax, next_idx, axis=1).reshape(-1)
        effective_n = mask.sum(-1).reshape(-1)

        return Transition(obs, act, next_obs, returns.reshape(-1), dones, effective_n)
