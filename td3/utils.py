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
    def __init__(self, n_env, buffer_size, n_obs, n_act, n_critic_obs=None, n_steps=1, gamma=0.99, device=None):
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs if n_critic_obs is not None else n_obs
        self.gamma = gamma
        self.n_steps = n_steps
        
        self.observations = jnp.zeros((n_env, buffer_size, n_obs))
        self.actions = jnp.zeros((n_env, buffer_size, n_act))
        self.rewards = jnp.zeros((n_env, buffer_size))
        self.dones = jnp.zeros((n_env, buffer_size), dtype=jnp.int32)
        self.truncations = jnp.zeros((n_env, buffer_size), dtype=jnp.int32)
        self.next_observations = jnp.zeros((n_env, buffer_size, n_obs))
        
        self.critic_observations = jnp.zeros((n_env, buffer_size, self.n_critic_obs))
        self.next_critic_observations = jnp.zeros((n_env, buffer_size, self.n_critic_obs))
        
        self.ptr = 0
        self.size = 0

    def add(self, transition):
        """Add transition dict with observations, actions, next observations, rewards, dones, truncations"""
        self.observations = self.observations.at[:, self.ptr].set(transition["observations"])
        self.actions = self.actions.at[:, self.ptr].set(transition["actions"])
        self.next_observations = self.next_observations.at[:, self.ptr].set(transition["next"]["observations"])
        self.rewards = self.rewards.at[:, self.ptr].set(transition["next"]["rewards"])
        self.dones = self.dones.at[:, self.ptr].set(transition["next"]["dones"])
        self.truncations = self.truncations.at[:, self.ptr].set(transition["next"]["truncations"])
        
        self.critic_observations = self.critic_observations.at[:, self.ptr].set(transition["critic_observations"])
        self.next_critic_observations = self.next_critic_observations.at[:, self.ptr].set(transition["next"]["critic_observations"])
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, rng: jax.random.PRNGKey, batch_size: int):
        """Sample batch_size transitions from the buffer with multi-step returns (JAX-optimized)"""
        total_samples = self.n_env * self.size
        flat_idx = jax.random.randint(rng, (batch_size,), 0, total_samples)
        
        env_idx = flat_idx // self.size
        buffer_idx = flat_idx % self.size
        
        obs = self.observations[env_idx, buffer_idx]
        actions = self.actions[env_idx, buffer_idx]
        
        step_range = jnp.arange(self.n_steps)  # [0, 1, 2, ..., n_steps-1]
        step_indices = (buffer_idx[:, None] + step_range[None, :]) % self.buffer_size  # (batch_size, n_steps)
        
        step_rewards = self.rewards[env_idx[:, None], step_indices]  # (batch_size, n_steps)
        step_dones = self.dones[env_idx[:, None], step_indices]  # (batch_size, n_steps)
        step_next_obs = self.next_observations[env_idx[:, None], step_indices]  # (batch_size, n_steps, obs_dim)
        
        gamma_powers = self.gamma ** step_range  # (n_steps,)
        
        cumulative_dones = jnp.concatenate([
            jnp.zeros((batch_size, 1), dtype=bool),
            jnp.cumsum(step_dones[:, :-1], axis=1) > 0
        ], axis=1)
        
        valid_mask = ~cumulative_dones  # (batch_size, n_steps)
        masked_rewards = jnp.where(valid_mask, step_rewards, 0.0)
        
        n_step_rewards = jnp.sum(masked_rewards * gamma_powers[None, :], axis=1)  # (batch_size,)
        
        effective_n = jnp.sum(valid_mask, axis=1)  # (batch_size,)
        
        last_valid_idx = jnp.maximum(effective_n - 1, 0)  # Ensure >= 0
        next_obs = step_next_obs[jnp.arange(batch_size), last_valid_idx]
        final_dones = step_dones[jnp.arange(batch_size), self.n_steps - 1]
        
        return {
            "observations": obs,
            "actions": actions,
            "next": {
                "observations": next_obs,
                "rewards": n_step_rewards,
                "dones": final_dones,
                "effective_n_steps": effective_n.astype(jnp.float32)
            }
        }
