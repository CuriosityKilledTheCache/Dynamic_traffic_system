import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_ml_training_data():
    """Load ML training data with time information."""
    try:
        episode_data = pd.read_csv('episode_results.csv')
        reward_data = pd.read_csv('reward_progress.csv')
        print("ML training data loaded successfully")
        return episode_data, reward_data
    except FileNotFoundError as e:
        print(f"Error loading ML training data: {e}")
        return None, None

def analyze_training_efficiency(episode_data, reward_data):
    """Analyze training efficiency over time."""
    if episode_data is None or reward_data is None:
        return
    
    # Calculate cumulative training time
    if 'EpisodeDuration' in episode_data.columns:
        episode_data['CumulativeTime'] = episode_data['EpisodeDuration'].cumsum()
        total_training_time = episode_data['CumulativeTime'].iloc[-1]
        training_efficiency = len(episode_data) / (total_training_time / 60)  # episodes per minute
    else:
        # Fallback if no duration data
        episode_data['CumulativeTime'] = np.arange(len(episode_data)) * 60  # Assume 60s per episode
        total_training_time = episode_data['CumulativeTime'].iloc[-1]
        training_efficiency = len(episode_data) / (total_training_time / 60)
    
    print(f"\nTraining Efficiency Analysis:")
    print(f"Total Training Time: {total_training_time/60:.2f} minutes")
    print(f"Total Episodes: {len(episode_data)}")
    print(f"Training Efficiency: {training_efficiency:.2f} episodes/minute")
    
    return total_training_time, training_efficiency

def create_training_progress_plots(episode_data, reward_data):
    """Create training progress plots with time on x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Agent Training Progress - Time-Based Analysis', fontsize=16)
    
    # Convert time to minutes for plotting
    time_minutes = episode_data['CumulativeTime'] / 60
    
    # Plot 1: Cumulative Reward over Time
    if 'CumulativeReward' in episode_data.columns:
        axes[0, 0].plot(time_minutes, episode_data['CumulativeReward'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Training Time (minutes)')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].set_title('Cumulative Reward Over Training Time')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Vehicle Throughput over Time
    if 'TotalVehicles' in episode_data.columns:
        # Calculate moving average for smoother visualization
        window = min(10, len(episode_data) // 10)
        if window > 1:
            smoothed_vehicles = episode_data['TotalVehicles'].rolling(window=window).mean()
            axes[0, 1].plot(time_minutes, smoothed_vehicles, 'g-', linewidth=2, label=f'Moving Avg (window={window})')
        axes[0, 1].plot(time_minutes, episode_data['TotalVehicles'], 'lightgreen', alpha=0.5, label='Raw Data')
        axes[0, 1].set_xlabel('Training Time (minutes)')
        axes[0, 1].set_ylabel('Total Vehicles per Episode')
        axes[0, 1].set_title('Vehicle Throughput During Training')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Fuel Consumption over Time
    if 'FuelConsumed' in episode_data.columns:
        window = min(10, len(episode_data) // 10)
        if window > 1:
            smoothed_fuel = episode_data['FuelConsumed'].rolling(window=window).mean()
            axes[1, 0].plot(time_minutes, smoothed_fuel, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
        axes[1, 0].plot(time_minutes, episode_data['FuelConsumed'], 'lightcoral', alpha=0.5, label='Raw Data')
        axes[1, 0].set_xlabel('Training Time (minutes)')
        axes[1, 0].set_ylabel('Fuel Consumed per Episode (L)')
        axes[1, 0].set_title('Fuel Consumption During Training')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate (Reward Improvement over Time)
    if 'CurrentReward' in episode_data.columns:
        # Calculate rate of reward improvement
        window = min(20, len(episode_data) // 5)
        if window > 1:
            reward_improvement = episode_data['CurrentReward'].rolling(window=window).mean().diff()
            valid_time = time_minutes[window:]
            valid_improvement = reward_improvement.dropna()
            if len(valid_improvement) > 0:
                axes[1, 1].plot(valid_time, valid_improvement, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Training Time (minutes)')
        axes[1, 1].set_ylabel('Reward Improvement Rate')
        axes[1, 1].set_title('Learning Rate (Reward Improvement)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_training_progress_time_based.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_training_summary(episode_data, reward_data, total_training_time, training_efficiency):
    """Generate training summary report."""
    summary = {
        'Training Metric': [
            'Total Training Time (minutes)',
            'Total Episodes Completed',
            'Training Efficiency (episodes/min)',
            'Final Cumulative Reward',
            'Average Vehicles per Episode',
            'Average Fuel per Episode (L)',
            'Best Episode Performance (Vehicles)',
            'Training Convergence Time (minutes)',
        ],
        'Value': []
    }
    
    # Fill in the values
    summary['Value'].append(f"{total_training_time/60:.2f}")
    summary['Value'].append(str(len(episode_data)))
    summary['Value'].append(f"{training_efficiency:.2f}")
    
    if 'CumulativeReward' in episode_data.columns:
        summary['Value'].append(f"{episode_data['CumulativeReward'].iloc[-1]:.2f}")
    else:
        summary['Value'].append("N/A")
    
    if 'TotalVehicles' in episode_data.columns:
        summary['Value'].append(f"{episode_data['TotalVehicles'].mean():.1f}")
        summary['Value'].append(f"{episode_data['TotalVehicles'].max()}")
    else:
        summary['Value'].append("N/A")
        summary['Value'].append("N/A")
    
    if 'FuelConsumed' in episode_data.columns:
        summary['Value'][5] = f"{episode_data['FuelConsumed'].mean():.2f}"
    else:
        summary['Value'].append("N/A")
    
    # Estimate convergence time (when reward stabilizes)
    if 'CumulativeReward' in episode_data.columns and len(episode_data) > 10:
        # Simple convergence detection: when reward increase rate drops below threshold
        reward_diff = episode_data['CumulativeReward'].diff().rolling(window=10).mean()
        convergence_idx = np.where(reward_diff < reward_diff.std() * 0.1)[0]
        if len(convergence_idx) > 0:
            convergence_time = episode_data.iloc[convergence_idx[0]]['CumulativeTime'] / 60
            summary['Value'].append(f"{convergence_time:.2f}")
        else:
            summary['Value'].append("Not converged")
    else:
        summary['Value'].append("N/A")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('ml_training_time_analysis.csv', index=False)
    
    print("\n" + "="*50)
    print("ML TRAINING TIME-BASED ANALYSIS SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)
    
    return summary_df

def main():
    """Main function for ML training time-based analysis."""
    print("Starting ML Training Time-Based Analysis...")
    print("="*50)
    
    # Load training data
    episode_data, reward_data = load_ml_training_data()
    
    if episode_data is None:
        return
    
    # Analyze training efficiency
    total_training_time, training_efficiency = analyze_training_efficiency(episode_data, reward_data)
    
    # Create training progress plots
    create_training_progress_plots(episode_data, reward_data)
    
    # Generate summary
    summary_df = generate_training_summary(episode_data, reward_data, total_training_time, training_efficiency)
    
    print(f"\nAnalysis complete! Files generated:")
    print(f"- ml_training_progress_time_based.png")
    print(f"- ml_training_time_analysis.csv")

if __name__ == "__main__":
    main()
