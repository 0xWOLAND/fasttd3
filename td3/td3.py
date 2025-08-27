from typing import Tuple, Callable
from flax.training.train_state import TrainState
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax

class DistributionalQNetwork(nn.Module):
    obs_dim: int
    act_dim: int
    num_atoms: int
    hidden_dim: int
    v_min: float
    v_max: float
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, act], axis=-1)
        x = self.activation(nn.Dense(self.hidden_dim)(x))
        x = self.activation(nn.Dense(self.hidden_dim // 2)(x))
        x = self.activation(nn.Dense(self.hidden_dim // 4)(x))
        logits = nn.Dense(self.num_atoms)(x)
        return logits  # raw logits; apply softmax outside if needed

    def support(self) -> jnp.ndarray:
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    def value(self, logits: jnp.ndarray) -> jnp.ndarray:
        probs = nn.softmax(logits, axis=-1)
        return jnp.sum(probs * self.support(), axis=-1) 

class Critic(nn.Module):
    obs_dim: int
    act_dim: int
    num_atoms: int
    hidden_dim: int
    v_min: float
    v_max: float

    def setup(self):
        self.qnet1 = DistributionalQNetwork(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            num_atoms=self.num_atoms,
            hidden_dim=self.hidden_dim,
            v_min=self.v_min,
            v_max=self.v_max,
        )
        self.qnet2 = DistributionalQNetwork(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            num_atoms=self.num_atoms,
            hidden_dim=self.hidden_dim,
            v_min=self.v_min,
            v_max=self.v_max,
        )
        # Cache support for efficiency
        self.support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: F821
        return self.qnet1(obs, act), self.qnet2(obs, act)

    def value(self, logits: jnp.ndarray) -> jnp.ndarray:
        probs = nn.softmax(logits, axis=-1)
        return jnp.sum(probs * self.support, axis=-1)

class Actor(nn.Module):
    obs_dim: int
    act_dim: int
    max_action: float
    hidden_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = self.activation(nn.Dense(self.hidden_dim)(obs))
        x = self.activation(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.act_dim)(x)
        return self.max_action * jnp.tanh(x)

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_def,
        critic_def,
        num_envs=1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        num_updates=1,
        seed=0,
    ):
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise  # Keep as ratio
        self.noise_clip = noise_clip  # Keep as ratio
        self.policy_freq = policy_freq
        self.num_updates = num_updates
        self.total_it = 0
        self.num_envs = num_envs

        self.rng = jax.random.PRNGKey(seed)
        dummy_obs = jnp.zeros((num_envs, state_dim))
        dummy_act = jnp.zeros((num_envs, action_dim))

        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        actor_params = actor_def.init(actor_key, dummy_obs)
        critic_params = critic_def.init(critic_key, dummy_obs, dummy_act)

        self.actor_def = actor_def
        self.critic_def = critic_def

        self.actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(actor_lr)
        )
        self.actor_target = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(actor_lr)
        )

        self.critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(critic_lr)
        )
        self.critic_target = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(critic_lr)
        )

        # Support for computing value from logits
        self.support = jnp.linspace(
            critic_def.v_min, critic_def.v_max, critic_def.num_atoms
        )

    def select_action(self, obs, add_noise=False):
        obs = jnp.asarray(obs)
        action = self.actor.apply_fn(self.actor.params, obs)
        
        if add_noise:
            self.rng, noise_key = jax.random.split(self.rng)
            noise = jax.random.normal(noise_key, action.shape) * 0.1 * self.max_action
            action = action + noise
            
        return jnp.clip(action, -self.max_action, self.max_action)

    def _compute_critic_loss(self, critic_params, target_critic_params, actor_target_params, batch, noise_key):
        def value_from_logits(logits):
            probs = nn.softmax(logits, axis=-1)
            return jnp.sum(probs * self.support, axis=-1)

        next_action = self.actor_target.apply_fn(actor_target_params, batch.next_obs)
        # Target policy smoothing with properly scaled noise
        noise = jax.random.normal(noise_key, next_action.shape) * self.policy_noise * self.max_action
        noise = jnp.clip(noise, -self.noise_clip * self.max_action, self.noise_clip * self.max_action)
        next_action = jnp.clip(
            next_action + noise, -self.max_action, self.max_action
        )

        target_q1_logits, target_q2_logits = self.critic_target.apply_fn(
            target_critic_params, batch.next_obs, next_action
        )
        target_q1 = value_from_logits(target_q1_logits)
        target_q2 = value_from_logits(target_q2_logits)
        target_q = jnp.minimum(target_q1, target_q2)
        target_q = batch.reward + (1.0 - batch.done) * (self.discount ** batch.effective_n) * target_q

        q1_logits, q2_logits = self.critic.apply_fn(
            critic_params, batch.obs, batch.action
        )
        q1 = value_from_logits(q1_logits)
        q2 = value_from_logits(q2_logits)

        loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return loss

    def _compute_actor_loss(self, actor_params, critic_params, batch):
        def value_from_logits(logits):
            probs = nn.softmax(logits, axis=-1)
            return jnp.sum(probs * self.support, axis=-1)

        actions = self.actor.apply_fn(actor_params, batch.obs)
        q1_logits, _ = self.critic.apply_fn(
            critic_params, batch.obs, actions
        )
        q1 = value_from_logits(q1_logits)
        return -q1.mean()

    def train(self, replay_buffer, batch_size):
        for _ in range(self.num_updates):
            self.total_it += 1
            self.rng, sample_key, noise_key = jax.random.split(self.rng, 3)
            batch = replay_buffer.sample(sample_key, batch_size)

            critic_loss_fn = jax.jit(lambda p: self._compute_critic_loss(p, self.critic_target.params, self.actor_target.params, batch, noise_key))
            critic_grads = jax.grad(critic_loss_fn)(self.critic.params)
            self.critic = self.critic.apply_gradients(grads=critic_grads)
            
            should_update_actor = (
                (self.num_updates > 1 and _ % self.policy_freq == 1) or
                (self.num_updates == 1 and self.total_it % self.policy_freq == 0)
            )
            
            if should_update_actor:
                actor_loss_fn = jax.jit(lambda p: self._compute_actor_loss(p, self.critic.params, batch))
                actor_grads = jax.grad(actor_loss_fn)(self.actor.params)
                self.actor = self.actor.apply_gradients(grads=actor_grads)

                self.actor_target = self._soft_update(self.actor_target, self.actor)
                self.critic_target = self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target: TrainState, source: TrainState):
        new_params = jax.tree_util.tree_map(
            lambda t, s: self.tau * s + (1 - self.tau) * t,
            target.params,
            source.params,
        )
        return target.replace(params=new_params)
