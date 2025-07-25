import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

def load_and_prepare_data():
    """Load CSV files and prepare data with time conversion."""
    try:
        # Load ML agent data
        ml_episode = pd.read_csv('episode_results.csv')
        ml_reward = pd.read_csv('reward_progress.csv')
        
        # Load static controller data
        static_episode = pd.read_csv('static_episode_results.csv')
        static_reward = pd.read_csv('static_reward_progress.csv')
        
        print("All CSV files loaded successfully")
        return ml_episode, ml_reward, static_episode, static_reward
    
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure you have run both ML and Static simulations and generated CSV files")
        return None, None, None, None

def convert_episodes_to_time(episode_df):
    """Convert episode-based data to time-based data."""
    if 'GreenLightTime' not in episode_df.columns:
        print("Warning: GreenLightTime column not found. Using default 60s per episode.")
        episode_df['GreenLightTime'] = 30.0
    
    # Calculate cumulative time
    episode_df['CumulativeTime'] = episode_df['GreenLightTime'].cumsum()
    
    # Create time arrays for interpolation
    time_points = np.arange(0, episode_df['CumulativeTime'].max(), 30)  # 30-second intervals
    
    return episode_df, time_points

def calculate_rates(episode_df, time_points):
    """Calculate rate-based metrics (per minute)."""
    # Interpolate metrics to regular time intervals
    interpolated_data = {}
    
    for column in ['TotalVehicles', 'VehiclesWaiting', 'FuelConsumed']:
        if column in episode_df.columns:
            # Calculate rates (per minute)
            rates = []
            for i in range(1, len(episode_df)):
                duration_minutes = episode_df.iloc[i]['GreenLightTime'] / 60.0
                if duration_minutes > 0:
                    rate = (episode_df.iloc[i][column] - episode_df.iloc[i-1][column]) / duration_minutes
                    rates.append(max(0, rate))  # Ensure non-negative rates
                else:
                    rates.append(0)
            
            # Add first episode rate
            if len(episode_df) > 0:
                first_rate = episode_df.iloc[0][column] / (episode_df.iloc[0]['GreenLightTime'] / 60.0) if episode_df.iloc[0]['GreenLightTime'] > 0 else 0
                rates.insert(0, first_rate)
            
            interpolated_data[f'{column}_Rate'] = np.interp(time_points, episode_df['CumulativeTime'], rates)
    
    return interpolated_data

def create_time_based_plots(ml_data, static_data, ml_time, static_time):
    """Create time-based comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML vs Static Traffic Signal Control - Time-Based Analysis', fontsize=16)
    
    # Plot 1: Vehicle Throughput Rate
    if 'TotalVehicles_Rate' in ml_data and 'TotalVehicles_Rate' in static_data:
        axes[0, 0].plot(ml_time/60, ml_data['TotalVehicles_Rate'], 'b-', label='ML Agent', linewidth=2)
        axes[0, 0].plot(static_time/60, static_data['TotalVehicles_Rate'], 'r-', label='Static Controller', linewidth=2)
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Vehicle Throughput (vehicles/min)')
        axes[0, 0].set_title('Vehicle Throughput Rate Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Queue Length Rate
    if 'VehiclesWaiting_Rate' in ml_data and 'VehiclesWaiting_Rate' in static_data:
        axes[0, 1].plot(ml_time/60, ml_data['VehiclesWaiting_Rate'], 'b-', label='ML Agent', linewidth=2)
        axes[0, 1].plot(static_time/60, static_data['VehiclesWaiting_Rate'], 'r-', label='Static Controller', linewidth=2)
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Queue Growth Rate (vehicles/min)')
        axes[0, 1].set_title('Queue Length Growth Rate Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Fuel Consumption Rate
    if 'FuelConsumed_Rate' in ml_data and 'FuelConsumed_Rate' in static_data:
        axes[1, 0].plot(ml_time/60, ml_data['FuelConsumed_Rate'], 'b-', label='ML Agent', linewidth=2)
        axes[1, 0].plot(static_time/60, static_data['FuelConsumed_Rate'], 'r-', label='Static Controller', linewidth=2)
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Fuel Consumption Rate (L/min)')
        axes[1, 0].set_title('Fuel Consumption Rate Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Performance Comparison
    ml_total_time = ml_time[-1] / 60 if len(ml_time) > 0 else 0
    static_total_time = static_time[-1] / 60 if len(static_time) > 0 else 0
    
    categories = ['Total Simulation\nTime (min)', 'Avg Throughput\n(veh/min)', 'Avg Queue Rate\n(veh/min)', 'Avg Fuel Rate\n(L/min)']
    ml_values = [
        ml_total_time,
        np.mean(ml_data.get('TotalVehicles_Rate', [0])),
        np.mean(ml_data.get('VehiclesWaiting_Rate', [0])),
        np.mean(ml_data.get('FuelConsumed_Rate', [0]))
    ]
    static_values = [
        static_total_time,
        np.mean(static_data.get('TotalVehicles_Rate', [0])),
        np.mean(static_data.get('VehiclesWaiting_Rate', [0])),
        np.mean(static_data.get('FuelConsumed_Rate', [0]))
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, ml_values, width, label='ML Agent', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, static_values, width, label='Static Controller', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Average Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_based_traffic_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_time_based_summary(ml_episode, static_episode, ml_data, static_data, ml_time, static_time):
    """Generate summary statistics for time-based analysis."""
    summary_data = {
        'Metric': [
            'Total Simulation Time (minutes)',
            'Average Throughput Rate (vehicles/min)',
            'Average Queue Growth Rate (vehicles/min)', 
            'Average Fuel Consumption Rate (L/min)',
            'ML vs Static Throughput Improvement (%)',
            'ML vs Static Queue Reduction (%)',
            'ML vs Static Fuel Savings (%)'
        ],
        'ML Agent': [
            f"{ml_time[-1]/60:.2f}" if len(ml_time) > 0 else "0.00",
            f"{np.mean(ml_data.get('TotalVehicles_Rate', [0])):.2f}",
            f"{np.mean(ml_data.get('VehiclesWaiting_Rate', [0])):.2f}",
            f"{np.mean(ml_data.get('FuelConsumed_Rate', [0])):.2f}",
            "-", "-", "-"
        ],
        'Static Controller': [
            f"{static_time[-1]/60:.2f}" if len(static_time) > 0 else "0.00",
            f"{np.mean(static_data.get('TotalVehicles_Rate', [0])):.2f}",
            f"{np.mean(static_data.get('VehiclesWaiting_Rate', [0])):.2f}",
            f"{np.mean(static_data.get('FuelConsumed_Rate', [0])):.2f}",
            "-", "-", "-"
        ]
    }
    
    # Calculate improvement percentages
    ml_throughput = np.mean(ml_data.get('TotalVehicles_Rate', [0]))
    static_throughput = np.mean(static_data.get('TotalVehicles_Rate', [0]))
    ml_queue = np.mean(ml_data.get('VehiclesWaiting_Rate', [0]))
    static_queue = np.mean(static_data.get('VehiclesWaiting_Rate', [0]))
    ml_fuel = np.mean(ml_data.get('FuelConsumed_Rate', [0]))
    static_fuel = np.mean(static_data.get('FuelConsumed_Rate', [0]))
    
    if static_throughput > 0:
        throughput_improvement = ((ml_throughput - static_throughput) / static_throughput) * 100
        summary_data['ML Agent'][4] = f"{throughput_improvement:.1f}%"
        summary_data['Static Controller'][4] = "0.0%"
    
    if static_queue > 0:
        queue_reduction = ((static_queue - ml_queue) / static_queue) * 100
        summary_data['ML Agent'][5] = f"{queue_reduction:.1f}%"
        summary_data['Static Controller'][5] = "0.0%"
    
    if static_fuel > 0:
        fuel_savings = ((static_fuel - ml_fuel) / static_fuel) * 100
        summary_data['ML Agent'][6] = f"{fuel_savings:.1f}%"
        summary_data['Static Controller'][6] = "0.0%"
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('time_based_comparison_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("TIME-BASED TRAFFIC SIGNAL ANALYSIS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    return summary_df

def main():
    """Main function to run time-based analysis."""
    print("Starting Time-Based Traffic Signal Analysis...")
    print("="*50)
    
    # Load data
    ml_episode, ml_reward, static_episode, static_reward = load_and_prepare_data()
    
    if ml_episode is None:
        return
    
    # Convert to time-based data
    ml_episode, ml_time = convert_episodes_to_time(ml_episode)
    static_episode, static_time = convert_episodes_to_time(static_episode)
    
    # Calculate rate-based metrics
    ml_data = calculate_rates(ml_episode, ml_time)
    static_data = calculate_rates(static_episode, static_time)
    
    # Create plots
    create_time_based_plots(ml_data, static_data, ml_time, static_time)
    
    # Generate summary
    summary_df = generate_time_based_summary(ml_episode, static_episode, ml_data, static_data, ml_time, static_time)
    
    print(f"\nAnalysis complete! Files generated:")
    print(f"- time_based_traffic_comparison.png")
    print(f"- time_based_comparison_summary.csv")

if __name__ == "__main__":
    main()
