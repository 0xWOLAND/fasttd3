import argparse
import pickle
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from td3 import Actor, Critic, TD3
from utils import ReplayBuffer

def create_networks(env):
    """Create actor and critic networks for the environment."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    actor_def = Actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        max_action=max_action,
        hidden_dim=256
    )
    
    critic_def = Critic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_atoms=51,
        hidden_dim=256,
        v_min=-200.0,
        v_max=200.0
    )
    
    return actor_def, critic_def, obs_dim, act_dim, max_action

def train_agent(args):
    """Train the TD3 agent and save the model."""
    env = gym.make(args.env)
    actor_def, critic_def, obs_dim, act_dim, max_action = create_networks(env)
    
    agent = TD3(
        state_dim=obs_dim,
        action_dim=act_dim,
        max_action=max_action,
        actor_def=actor_def,
        critic_def=critic_def,
        seed=args.seed
    )
    
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=args.buffer_size,
        n_env=1,
        n_steps=args.n_steps,
        gamma=args.gamma
    )
    
    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    # Early stopping variables
    recent_rewards = []
    eval_env = gym.make(args.env)
    
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = np.array(agent.select_action(obs[None], add_noise=True))[0]
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        replay_buffer.add(
            obs[None], action[None], next_obs[None], 
            np.array([reward]), np.array([float(done)])
        )
        
        obs = next_obs
        episode_reward += reward
        
        if t >= args.start_timesteps:
            agent.train(replay_buffer, args.batch_size)
        
        if done:
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > args.eval_window:
                recent_rewards.pop(0)
            
            avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward
            print(f"Episode {episode_num + 1}: Reward: {episode_reward:.2f}, Avg({len(recent_rewards)}): {avg_reward:.2f}")
            
            # Check if solved
            if len(recent_rewards) >= args.eval_window and avg_reward >= args.solve_threshold:
                print(f"Environment solved! Average reward {avg_reward:.2f} >= {args.solve_threshold}")
                break
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
    
    # Save the trained model
    model_data = {
        'actor_params': agent.actor.params,
        'critic_params': agent.critic.params,
        'actor_def': actor_def,
        'critic_def': critic_def,
        'max_action': max_action,
        'obs_dim': obs_dim,
        'act_dim': act_dim
    }
    
    with open(args.save_model, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {args.save_model}")
    env.close()
    eval_env.close()

def visualize_agent(args):
    """Load and visualize the trained agent."""
    try:
        with open(args.load_model, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Model file {args.load_model} not found. Please train a model first.")
        return
    
    env = gym.make(args.env, render_mode="human")
    
    # Recreate the agent with saved parameters
    agent = TD3(
        state_dim=model_data['obs_dim'],
        action_dim=model_data['act_dim'],
        max_action=model_data['max_action'],
        actor_def=model_data['actor_def'],
        critic_def=model_data['critic_def']
    )
    
    # Load the trained parameters
    agent.actor = agent.actor.replace(params=model_data['actor_params'])
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = np.array(agent.select_action(obs[None], add_noise=False))[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            env.render()
        
        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description="TD3 Training and Visualization")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--visualize", action="store_true", help="Visualize the trained agent")
    
    args = parser.parse_args()
    
    # Set default configuration
    args.env = "Humanoid-v5"
    args.seed = 0
    args.max_timesteps = 10000000
    args.start_timesteps = 25000
    args.batch_size = 256
    args.buffer_size = 1000000
    args.n_steps = 3
    args.gamma = 0.99
    args.save_model = "td3_humanoid_model.pkl"
    args.load_model = "td3_humanoid_model.pkl"
    args.num_episodes = 10
    args.solve_threshold = 5000.0
    args.eval_window = 100
    
    if args.train:
        train_agent(args)
    elif args.visualize:
        visualize_agent(args)
    else:
        print("Please specify either --train or --visualize")
        parser.print_help()

if __name__ == "__main__":
    main()