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
        # Use JAX arrays on GPU for storage to avoid CPU/GPU transfers
        self.obs = jnp.zeros((n_env, size, obs_dim), dtype=jnp.float32)
        self.next_obs = jnp.zeros((n_env, size, obs_dim), dtype=jnp.float32)
        self.action = jnp.zeros((n_env, size, act_dim), dtype=jnp.float32)
        self.reward = jnp.zeros((n_env, size), dtype=jnp.float32)
        self.done = jnp.zeros((n_env, size), dtype=jnp.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.n_env = n_env
        self.n_steps = n_steps
        self.gamma = gamma

    def add(self, obs, action, next_obs, reward, done):
        # Store directly as JAX arrays (no CPU conversion)
        self.obs = self.obs.at[:, self.ptr].set(obs)
        self.action = self.action.at[:, self.ptr].set(action)
        self.next_obs = self.next_obs.at[:, self.ptr].set(next_obs)
        self.reward = self.reward.at[:, self.ptr].set(reward)
        self.done = self.done.at[:, self.ptr].set(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: jax.random.PRNGKey, batch_size: int) -> Transition:
        if self.size <= self.n_steps:
            raise ValueError(f"Not enough samples in buffer: {self.size} <= {self.n_steps}")
        
        # Convert JAX key to numpy seed
        rng_np = np.random.RandomState(int(rng[0]))
        
        # Per-environment sampling + buffer safety
        batch_per_env = batch_size // self.n_env
        max_idx = self.size - self.n_steps if self.size < self.max_size else self.max_size - self.n_steps
        idx = rng_np.randint(0, max_idx, (self.n_env, batch_per_env))
        
        # N-step indices with wraparound
        steps = np.arange(self.n_steps)[None, None, :]
        all_idx = (idx[..., None] + steps) % self.max_size
        
        # Sample data efficiently using numpy
        env_grid = np.arange(self.n_env)[:, None, None]
        rewards = self.reward[env_grid, all_idx]
        dones = self.done[env_grid, all_idx]
        
        # Compute returns with episode boundaries
        mask = np.cumprod(1.0 - np.pad(dones[..., :-1], ((0, 0), (0, 0), (1, 0))), axis=-1)
        returns = np.sum(rewards * mask * (self.gamma ** steps), axis=-1)
        
        # Find terminal states for next_obs
        first_done = np.argmax(dones, axis=-1)
        no_done = dones.sum(-1) == 0
        final_idx = np.where(no_done, self.n_steps - 1, first_done)
        next_idx = all_idx[np.arange(self.n_env)[:, None], np.arange(batch_per_env)[None, :], final_idx]
        
        # Gather observations using numpy
        obs = self.obs[env_grid[:, :, 0], idx].reshape(-1, self.obs.shape[-1])
        act = self.action[env_grid[:, :, 0], idx].reshape(-1, self.action.shape[-1])
        next_obs = self.next_obs[env_grid[:, :, 0], next_idx].reshape(-1, self.obs.shape[-1])
        final_done = self.done[env_grid[:, :, 0], next_idx].reshape(-1)
        
        # Return JAX arrays directly (already on GPU)
        return Transition(
            obs, 
            act, 
            next_obs, 
            returns.reshape(-1), 
            final_done, 
            mask.sum(-1).reshape(-1)
        )
