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

    def sample(self, batch_size: int) -> Transition:
        idx = jax.random.randint(jax.random.PRNGKey(0), (self.n_env, batch_size), 0, self.size - self.n_steps)
        steps = jnp.arange(self.n_steps)[None, None, :]
        all_idx = (idx[..., None] + steps) % self.max_size  # [env, B, n_step]

        rewards = jnp.take_along_axis(self.reward, all_idx, axis=1)  # [env, B, n_step]
        dones = jnp.take_along_axis(self.done, all_idx, axis=1)

        mask = jnp.cumprod(1.0 - jnp.pad(dones[..., :-1], ((0, 0), (0, 0), (1, 0))), axis=-1)
        discounted = rewards * (mask * self.gamma ** steps)
        returns = discounted.sum(-1)

        next_idx = jnp.take_along_axis(all_idx, jnp.argmin((dones > 0), axis=-1, keepdims=True), axis=-1).squeeze(-1)
        no_done = dones.sum(-1) == 0
        next_idx = jnp.where(no_done, all_idx[..., -1], next_idx)

        obs = jnp.take_along_axis(self.obs, idx[..., None], axis=1).reshape(-1, self.obs.shape[-1])
        act = jnp.take_along_axis(self.action, idx[..., None], axis=1).reshape(-1, self.action.shape[-1])
        next_obs = jnp.take_along_axis(self.next_obs, next_idx[..., None], axis=1).reshape(-1, self.obs.shape[-1])
        dones = jnp.take_along_axis(self.done, next_idx, axis=1).reshape(-1)
        effective_n = mask.sum(-1).reshape(-1)

        return Transition(obs, act, next_obs, returns.reshape(-1), dones, effective_n)