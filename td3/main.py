import argparse
import pickle
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import imageio
from td3 import Actor, Critic, TD3
from utils import ReplayBuffer

def train_agent(args):
    """Train the TD3 agent and save the model."""
    envs = gym.make_vec(args.env, args.num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    max_action = float(envs.single_action_space.high[0])
    
    actor_def = Actor(obs_dim=obs_dim, act_dim=act_dim, max_action=max_action, hidden_dim=256)
    critic_def = Critic(obs_dim=obs_dim, act_dim=act_dim, num_atoms=51, hidden_dim=256, v_min=-20.0, v_max=5.0)
    
    agent = TD3(
        state_dim=obs_dim, action_dim=act_dim, max_action=max_action,
        actor_def=actor_def, critic_def=critic_def, num_envs=args.num_envs, seed=args.seed
    )
    
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=args.buffer_size,
        n_env=args.num_envs, n_steps=args.n_steps, gamma=args.gamma
    )
    
    obs, _ = envs.reset(seed=args.seed)
    episode_rewards = np.zeros(args.num_envs)
    episode_num = 0
    recent_rewards = []
    
    print(envs)
    for t in range(args.max_timesteps):
        actions = envs.action_space.sample() if t < args.start_timesteps else agent.select_action(obs, add_noise=True)
        next_obs, rewards, dones, truncated, _ = envs.step(actions)
        dones = dones | truncated
        
        replay_buffer.add(obs, actions, next_obs, rewards, dones.astype(float))
        obs = next_obs
        episode_rewards += rewards
        
        if t >= args.start_timesteps:
            agent.train(replay_buffer, args.batch_size)
        
        if dones.any():
            for i in np.where(dones)[0]:
                recent_rewards.append(episode_rewards[i])
                if len(recent_rewards) > args.eval_window:
                    recent_rewards.pop(0)
                
                avg_reward = np.mean(recent_rewards) if recent_rewards else episode_rewards[i]
                print(f"Episode {episode_num + 1}: Reward: {episode_rewards[i]:.2f}, Avg({len(recent_rewards)}): {avg_reward:.2f}")
                
                if (episode_num + 1) % 20 == 0:
                    model_data = {'actor_params': agent.actor.params, 'critic_params': agent.critic.params, 
                                 'max_action': max_action, 'obs_dim': obs_dim, 'act_dim': act_dim}
                    with open(args.save_model, 'wb') as f:
                        pickle.dump(model_data, f)
                
                if len(recent_rewards) >= args.eval_window and avg_reward >= args.solve_threshold:
                    print(f"Environment solved! Average reward {avg_reward:.2f} >= {args.solve_threshold}")
                    envs.close()
                    return
                
                episode_num += 1
            
            episode_rewards[dones] = 0
    
    # Save the trained model
    model_data = {
        'actor_params': agent.actor.params,
        'critic_params': agent.critic.params,
        'max_action': max_action,
        'obs_dim': obs_dim,
        'act_dim': act_dim
    }
    
    with open(args.save_model, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {args.save_model}")
    envs.close()

def visualize_agent(args):
    """Load and visualize the trained agent."""
    try:
        with open(args.load_model, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Model file {args.load_model} not found. Please train a model first.")
        return
    
    # Create environments for both visualization and GIF
    env_visual = gym.make(args.env, render_mode="human")
    env_gif = gym.make(args.env, render_mode="rgb_array")
    
    actor_def = Actor(obs_dim=model_data['obs_dim'], act_dim=model_data['act_dim'], 
                     max_action=model_data['max_action'], hidden_dim=256)
    critic_def = Critic(obs_dim=model_data['obs_dim'], act_dim=model_data['act_dim'], 
                       num_atoms=51, hidden_dim=256, v_min=-20.0, v_max=5.0)
    
    agent = TD3(
        state_dim=model_data['obs_dim'],
        action_dim=model_data['act_dim'],
        max_action=model_data['max_action'],
        actor_def=actor_def,
        critic_def=critic_def
    )
    
    # Load the trained parameters
    agent.actor = agent.actor.replace(params=model_data['actor_params'])
    
    frames = [] if args.gif else None
    
    for episode in range(args.num_episodes):
        obs_visual, _ = env_visual.reset()
        obs_gif, _ = env_gif.reset() if args.gif else (None, None)
        episode_reward = 0
        done = False
        
        while not done:
            action = np.array(agent.select_action(obs_visual[None], add_noise=False))[0]
            
            # Step both environments
            obs_visual, reward, terminated, truncated, _ = env_visual.step(action)
            done = terminated or truncated
            episode_reward += reward
            env_visual.render()
            
            if args.gif:
                obs_gif, _, _, _, _ = env_gif.step(action)
                frames.append(env_gif.render())
        
        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}")
    
    env_visual.close()
    if args.gif:
        env_gif.close()
        gif_filename = f"{args.env}_agent.gif"
        imageio.mimsave(gif_filename, frames, fps=30)
        print(f"Saved visualization as {gif_filename}")

def main():
    parser = argparse.ArgumentParser(description="TD3 Training and Visualization")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--visualize", action="store_true", help="Visualize the trained agent")
    parser.add_argument("--gif", action="store_true", help="Save visualization as GIF")
    
    args = parser.parse_args()
    
    # Set default configuration for Pendulum-v1
    args.env = "Pendulum-v1"
    args.seed = 0
    args.max_timesteps = 50000
    args.start_timesteps = 10000
    args.batch_size = 512
    args.buffer_size = 1000000
    args.n_steps = 1
    args.gamma = 0.99
    args.save_model = "td3_pendulum_model.pkl"
    args.load_model = "td3_pendulum_model.pkl"
    args.num_episodes = 10
    args.solve_threshold = -200.0
    args.eval_window = 100
    args.num_envs = 16
    
    if args.train:
        train_agent(args)
    elif args.visualize:
        visualize_agent(args)
    else:
        print("Please specify either --train or --visualize")
        parser.print_help()

if __name__ == "__main__":
    main()