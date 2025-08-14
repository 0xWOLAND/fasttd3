#!/usr/bin/env python3
import gymnasium as gym
import numpy as np

def debug_environment():
    """Debug the Humanoid-v4 environment and random policy performance"""
    
    # Single environment for evaluation
    env = gym.make("Humanoid-v5", render_mode="human")
    
    print("=== Environment Debug ===")
    print(f"Action space: {env.action_space}")
    print(f"Action bounds: {env.action_space.low} to {env.action_space.high}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max episode steps: {env._max_episode_steps}")
    
    print(f"\nRandom action sample: {env.action_space.sample()}")
    
    # Test random policy performance
    print("\n=== Random Policy Evaluation ===")
    rewards = []
    episode_lengths = []
    
    for episode in range(5):
        obs, _ = env.reset(seed=episode + 100)  # Same seed pattern as your eval
        episode_reward = 0
        step_count = 0
        max_steps = 1000
        
        print(f"\nEpisode {episode + 1}:")
        
        while step_count < max_steps:
            # Random action (same as your training during random phase)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Render the environment
            env.render()
            
            # Print first few rewards to see what's happening
            if step_count <= 10:
                print(f"  Step {step_count}: reward={reward:.3f}, cumulative={episode_reward:.3f}")
            
            # Small delay for better visualization
            import time
            time.sleep(0.02)
            
            done = terminated or truncated
            if done:
                print(f"  Episode ended at step {step_count} (terminated={terminated}, truncated={truncated})")
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"  Final reward: {episode_reward:.1f}, length: {step_count}")
        
        if episode < 4:  # Pause between episodes except for last one
            input("Press Enter for next episode...")
    
    avg_reward = np.mean(rewards)
    print(f"\n=== Results ===")
    print(f"Individual rewards: {[f'{r:.1f}' for r in rewards]}")
    print(f"Average reward: {avg_reward:.1f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"Expected for random policy: ~5-20")
    
    if avg_reward > 100:
        print(f"\n⚠️  WARNING: Random policy reward {avg_reward:.1f} is suspiciously high!")
        print("This suggests a potential issue with:")
        print("- Environment setup")
        print("- Reward function")
        print("- Action space configuration")
    else:
        print(f"\n✅ Random policy performance looks normal")
    
    env.close()

if __name__ == "__main__":
    debug_environment()