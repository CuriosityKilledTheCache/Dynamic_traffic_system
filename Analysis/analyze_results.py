import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_traffic_results():
    # File paths (adjust based on where Unity saves them)
    episode_file = "episode_results.csv"
    reward_file = "reward_progress.csv"
    
    # Check if files exist
    if not os.path.exists(episode_file):
        print(f"Episode file not found: {episode_file}")
        print(f"Check Unity's persistentDataPath folder")
        return
        
    if not os.path.exists(reward_file):
        print(f"Reward file not found: {reward_file}")
        return
    
    # Load data
    episode_df = pd.read_csv(episode_file)
    reward_df = pd.read_csv(reward_file)
    
    # Print basic statistics
    print("=== EPISODE STATISTICS ===")
    print(f"Total Episodes: {len(episode_df)}")
    print(f"Average Vehicles per Episode: {episode_df['TotalVehicles'].mean():.2f}")
    print(f"Average Wait Time: {episode_df['VehiclesWaiting'].mean():.2f}")
    print(f"Average Episode Duration: {episode_df['EpisodeDuration'].mean():.2f} seconds")
    print(f"Final Cumulative Reward: {episode_df['CumulativeReward'].iloc[-1]:.2f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Traffic Signal ML Training Results', fontsize=16)
    
    # 1. Vehicles Served Over Time
    axes[0,0].plot(episode_df['Episode'], episode_df['TotalVehicles'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Vehicles Served')
    axes[0,0].set_title('Vehicle Throughput Over Training')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Vehicles Waiting Over Time
    axes[0,1].plot(episode_df['Episode'], episode_df['VehiclesWaiting'], 'r-', linewidth=2)
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Vehicles Waiting')
    axes[0,1].set_title('Queue Length Over Training')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Cumulative Reward
    axes[1,0].plot(episode_df['Episode'], episode_df['CumulativeReward'], 'g-', linewidth=2)
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Cumulative Reward')
    axes[1,0].set_title('Learning Progress (Cumulative Reward)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Episode Duration Distribution
    axes[1,1].hist(episode_df['EpisodeDuration'], bins=20, alpha=0.7, color='purple')
    axes[1,1].set_xlabel('Episode Duration (seconds)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Episode Duration Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('traffic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create reward progress chart
    if len(reward_df) > 100:  # Only if we have enough data points
        plt.figure(figsize=(12, 6))
        plt.plot(reward_df['Step'], reward_df['Reward'], alpha=0.3, color='blue', label='Step Reward')
        
        # Add moving average
        window_size = max(1, len(reward_df) // 50)
        moving_avg = reward_df['Reward'].rolling(window=window_size).mean()
        plt.plot(reward_df['Step'], moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Reward')
        plt.title('Reward Progress During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('reward_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Generate summary table
    summary_stats = {
        'Metric': [
            'Total Episodes',
            'Avg Vehicles/Episode', 
            'Avg Vehicles Waiting',
            'Avg Episode Duration',
            'Final Cumulative Reward',
            'Best Episode (Vehicles)',
            'Worst Episode (Vehicles)'
        ],
        'Value': [
            len(episode_df),
            f"{episode_df['TotalVehicles'].mean():.2f}",
            f"{episode_df['VehiclesWaiting'].mean():.2f}",
            f"{episode_df['EpisodeDuration'].mean():.2f} sec",
            f"{episode_df['CumulativeReward'].iloc[-1]:.2f}",
            f"{episode_df['TotalVehicles'].max():.0f}",
            f"{episode_df['TotalVehicles'].min():.0f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    print("\n=== SUMMARY TABLE ===")
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv('training_summary.csv', index=False)
    print(f"\nFiles generated:")
    print("- traffic_analysis.png")
    print("- reward_progress.png") 
    print("- training_summary.csv")

if __name__ == "__main__":
    analyze_traffic_results()
