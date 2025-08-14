#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import argparse
from pathlib import Path

def parse_log_file(log_file):
    """Parse training log file and extract metrics"""
    timesteps = []
    episodes = []
    avg_rewards = []
    best_rewards = []
    worst_rewards = []
    evaluations = []
    eval_timesteps = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse training progress: "T: 12345 | Episodes: 123 | Avg(50): 123.4 | Best(50): 234.5 | Worst(50): 12.3"
            match = re.search(r'T:\s*(\d+)\s*\|\s*Episodes:\s*(\d+)\s*\|\s*Avg\(50\):\s*([\d.-]+)\s*\|\s*Best\(50\):\s*([\d.-]+)\s*\|\s*Worst\(50\):\s*([\d.-]+)', line)
            if match:
                timestep = int(match.group(1))
                episode_count = int(match.group(2))
                avg_reward = float(match.group(3))
                best_reward = float(match.group(4))
                worst_reward = float(match.group(5))
                
                # Debug: print parsed values occasionally
                if timestep % 500 == 0:
                    print(f"Parsed T:{timestep}, Avg:{avg_reward}, Best:{best_reward}")
                
                timesteps.append(timestep)
                episodes.append(episode_count)
                avg_rewards.append(avg_reward)
                best_rewards.append(best_reward)
                worst_rewards.append(worst_reward)
            
            # Parse evaluation: "Evaluation: 123.4"
            match = re.search(r'Evaluation:\s*([\d.-]+)', line)
            if match:
                eval_reward = float(match.group(1))
                evaluations.append(eval_reward)
                # Estimate timestep based on evaluation frequency (every 5000 steps)
                eval_timesteps.append(len(evaluations) * 5000)
    
    print(f"Parsed {len(timesteps)} training entries, {len(evaluations)} evaluations")
    if timesteps:
        print(f"Timestep range: {min(timesteps)} - {max(timesteps)}")
        print(f"Reward range: {min(avg_rewards):.1f} - {max(avg_rewards):.1f}")
    
    return {
        'timesteps': timesteps,
        'episodes': episodes,
        'avg_rewards': avg_rewards,
        'best_rewards': best_rewards,
        'worst_rewards': worst_rewards,
        'evaluations': evaluations,
        'eval_timesteps': eval_timesteps
    }

def plot_training_progress(data, save_path=None, title="FastTD3 Training Progress"):
    """Create comprehensive training plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Average Reward vs Timesteps
    if data['timesteps'] and data['avg_rewards']:
        ax1.plot(data['timesteps'], data['avg_rewards'], 'b-', linewidth=2, label='Avg(50)')
        ax1.fill_between(data['timesteps'], data['worst_rewards'], data['best_rewards'], 
                        alpha=0.3, color='blue', label='Best/Worst(50)')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation Rewards
    if data['eval_timesteps'] and data['evaluations']:
        ax2.plot(data['eval_timesteps'], data['evaluations'], 'ro-', linewidth=2, markersize=4)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Evaluation Reward')
        ax2.set_title('Evaluation Performance')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined Training + Evaluation
    if data['timesteps'] and data['avg_rewards']:
        ax3.plot(data['timesteps'], data['avg_rewards'], 'b-', linewidth=2, alpha=0.7, label='Training Avg(50)')
    if data['eval_timesteps'] and data['evaluations']:
        ax3.plot(data['eval_timesteps'], data['evaluations'], 'ro-', linewidth=2, markersize=6, label='Evaluation')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Training vs Evaluation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        # Auto-save with default name
        default_path = f"training_plot.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {default_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot FastTD3 training logs')
    parser.add_argument('log_file', help='Log file to plot')
    
    args = parser.parse_args()
    
    data = parse_log_file(args.log_file)
    plot_training_progress(data, title=f"Training Progress - {Path(args.log_file).stem}")

if __name__ == "__main__":
    main()