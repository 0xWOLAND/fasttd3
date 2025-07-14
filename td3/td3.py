from ast import Tuple
from flax.train_state import TrainState
import flax.linen as nn
from typing import Callable
import jax.numpy as jnp
import jax

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

    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: F821
        return self.qnet1(obs, act), self.qnet2(obs, act)

    def value(self, logits: jnp.ndarray) -> jnp.ndarray:
        support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        probs = nn.softmax(logits, axis=-1)
        return jnp.sum(probs * support, axis=-1)

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
        seed=0,
    ):
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
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

    def select_action(self, obs):
        obs = jnp.asarray(obs)
        action = self.actor.apply_fn(self.actor.params, obs)
        return jnp.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        batch = replay_buffer.sample(batch_size)
        self.rng, noise_key = jax.random.split(self.rng)

        def value_from_logits(logits):
            probs = nn.softmax(logits, axis=-1)
            return jnp.sum(probs * self.support, axis=-1)

        def critic_loss_fn(critic_params):
            next_action = self.actor_target.apply_fn(
                self.actor_target.params, batch.next_obs
            )
            noise = jnp.clip(
                jax.random.normal(noise_key, next_action.shape) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip,
            )
            next_action = jnp.clip(
                next_action + noise, -self.max_action, self.max_action
            )

            target_q1_logits, target_q2_logits = self.critic_target.apply_fn(
                self.critic_target.params, batch.next_obs, next_action
            )
            target_q1 = value_from_logits(target_q1_logits)
            target_q2 = value_from_logits(target_q2_logits)
            target_q = jnp.minimum(target_q1, target_q2)
            target_q = batch.reward + (1.0 - batch.done) * self.discount * target_q

            q1_logits, q2_logits = self.critic.apply_fn(
                critic_params, batch.obs, batch.action
            )
            q1 = value_from_logits(q1_logits)
            q2 = value_from_logits(q2_logits)

            loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return loss

        critic_grads = jax.grad(critic_loss_fn)(self.critic.params)
        self.critic = self.critic.apply_gradients(grads=critic_grads)

        if self.total_it % self.policy_freq == 0:

            def actor_loss_fn(actor_params):
                actions = self.actor.apply_fn(actor_params, batch.obs)
                q1_logits, _ = self.critic.apply_fn(
                    self.critic.params, batch.obs, actions
                )
                q1 = value_from_logits(q1_logits)
                return -q1.mean()

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
