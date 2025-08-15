import numpy as np
import jax.numpy as jnp


class EmpiricalNormalization:
    """FastTD3-style observation normalization using Chan's parallel algorithm"""
    def __init__(self, shape, eps=1e-2):
        self.eps = eps
        self.mean = jnp.zeros((1, shape))
        self.var = jnp.ones((1, shape))
        self.count = 0
    
    def normalize(self, obs):
        obs = jnp.asarray(obs)
        std = jnp.sqrt(self.var + self.eps)
        return (obs - self.mean) / std
    
    def update(self, obs):
        obs = jnp.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        batch_size = obs.shape[0]
        batch_mean = jnp.mean(obs, axis=0, keepdims=True)
        new_count = self.count + batch_size
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = jnp.var(obs, axis=0, keepdims=True)
        else:
            # Chan's parallel algorithm
            delta = batch_mean - self.mean
            self.mean = self.mean + (batch_size / new_count) * delta
            
            batch_var = jnp.var(obs, axis=0, keepdims=True)
            delta2 = batch_mean - self.mean
            m_a = self.var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self.var = M2 / new_count
        
        self.count = new_count


class RewardNormalizer:
    """Simple reward normalization"""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
    
    def normalize(self, reward):
        reward = jnp.asarray(reward)
        return (reward - self.mean) / (jnp.sqrt(self.var) + 1e-8)
    
    def update(self, rewards):
        rewards = np.asarray(rewards)
        batch_count = len(rewards)
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        update_rate = min(0.01, batch_count / (self.count + batch_count))
        self.mean = (1 - update_rate) * self.mean + update_rate * batch_mean
        self.var = (1 - update_rate) * self.var + update_rate * batch_var
        self.count += batch_count