import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_comparison_results():
    # File paths for both ML and static results
    ml_episode_file = "episode_results.csv"
    ml_reward_file = "reward_progress.csv"
    static_episode_file = "static_episode_results.csv"
    static_reward_file = "static_reward_progress.csv"
    
    # Load ML data
    ml_episode_df = pd.read_csv(ml_episode_file) if os.path.exists(ml_episode_file) else None
    ml_reward_df = pd.read_csv(ml_reward_file) if os.path.exists(ml_reward_file) else None
    
    # Load Static data
    static_episode_df = pd.read_csv(static_episode_file) if os.path.exists(static_episode_file) else None
    static_reward_df = pd.read_csv(static_reward_file) if os.path.exists(static_reward_file) else None
    
    # -------------- New: Check that FuelConsumed exists --------------
    if ml_episode_df is not None:
        if "FuelConsumed" not in ml_episode_df.columns:
            raise KeyError("episode_results.csv missing ‘FuelConsumed’ column")
    if static_episode_df is not None:
        if "FuelConsumed" not in static_episode_df.columns:
            raise KeyError("static_episode_results.csv missing ‘FuelConsumed’ column")
            
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML vs Static Traffic Signal Control Comparison', fontsize=16)
    
    # 1. Vehicle Throughput Comparison
    if ml_episode_df is not None and static_episode_df is not None:
        axes[0,0].plot(ml_episode_df['Episode'], ml_episode_df['TotalVehicles'], 
                      'b-', linewidth=2, label='ML Agent')
        axes[0,0].plot(static_episode_df['Episode'], static_episode_df['TotalVehicles'], 
                      'r-', linewidth=2, label='Fixed-Time')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Vehicles Served')
        axes[0,0].set_title('Vehicle Throughput Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Queue Length Comparison
    if ml_episode_df is not None and static_episode_df is not None:
        axes[0,1].plot(ml_episode_df['Episode'], ml_episode_df['VehiclesWaiting'], 
                      'b-', linewidth=2, label='ML Agent')
        axes[0,1].plot(static_episode_df['Episode'], static_episode_df['VehiclesWaiting'], 
                      'r-', linewidth=2, label='Fixed-Time')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Vehicles Waiting')
        axes[0,1].set_title('Queue Length Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # # 3. Average Wait Time Comparison
    # if ml_episode_df is not None and static_episode_df is not None:
    #     # Assuming you add AverageWaitTime to ML logging as well
    #     if 'AverageWaitTime' in ml_episode_df.columns and 'AverageWaitTime' in static_episode_df.columns:
    #         axes[1,0].boxplot([ml_episode_df['AverageWaitTime'], static_episode_df['AverageWaitTime']], 
    #                         labels=['ML Agent', 'Fixed-Time'])
    #         axes[1,0].set_ylabel('Average Wait Time (seconds)')
    #         axes[1,0].set_title('Wait Time Distribution Comparison')
    #         axes[1,0].grid(True, alpha=0.3)

    # 3. **Fuel Consumption Comparison** (new subplot)
    if ml_episode_df is not None and static_episode_df is not None:
        axes[1,0].plot(ml_episode_df["Episode"],   ml_episode_df["FuelConsumed"],   'g-', linewidth=2, label="ML Agent")
        axes[1,0].plot(static_episode_df["Episode"], static_episode_df["FuelConsumed"], 'm--', linewidth=2, label="Fixed-Time")
        axes[1,0].set_xlabel("Episode")
        axes[1,0].set_ylabel("Fuel Consumed (L)")
        axes[1,0].set_title("Fuel Consumption per Episode")
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

    
    # 4. Performance Summary
    if ml_episode_df is not None and static_episode_df is not None:
        ml_avg_vehicles = ml_episode_df['TotalVehicles'].mean()
        static_avg_vehicles = static_episode_df['TotalVehicles'].mean()
        
        # **new** fuel averages
        ml_avg_fuel = ml_episode_df["FuelConsumed"].mean()
        static_avg_fuel = static_episode_df["FuelConsumed"].mean()

        categories = ['Avg Vehicles\nServed', 'Avg Queue\nLength', 'Avg Fuel\nConsumed (L)']
        ml_values = [ml_avg_vehicles, ml_episode_df['VehiclesWaiting'].mean(), ml_avg_fuel]

        static_values = [static_avg_vehicles, static_episode_df['VehiclesWaiting'].mean(), static_avg_fuel]
        
        x = np.arange(len(categories))
        width = 0.3
        
        axes[1,1].bar(x - width/2, ml_values, width, label='ML Agent', color='blue', alpha=0.7)
        axes[1,1].bar(x + width/2, static_values, width, label='Fixed-Time', color='red', alpha=0.7)
        
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Performance Summary')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(categories)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_vs_static_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate comparison table
    if ml_episode_df is not None and static_episode_df is not None:
        comparison_data = {
            'Metric': [
                'Average Vehicles Served',
                'Average Queue Length', 
                'Average Fuel Consumed (L)',
                'Max Throughput',
                'Min Queue Length',
                'Performance Improvement (%)'
            ],
            'ML Agent': [
                f"{ml_episode_df['TotalVehicles'].mean():.2f}",
                f"{ml_episode_df['VehiclesWaiting'].mean():.2f}",
                f"{ml_avg_fuel:.2f}",
                f"{ml_episode_df['TotalVehicles'].max():.0f}",
                f"{ml_episode_df['VehiclesWaiting'].min():.0f}",
                "-"
            ],
            'Fixed-Time': [
                f"{static_episode_df['TotalVehicles'].mean():.2f}",
                f"{static_episode_df['VehiclesWaiting'].mean():.2f}",
                f"{static_avg_fuel:.2f}",
                f"{static_episode_df['TotalVehicles'].max():.0f}",
                f"{static_episode_df['VehiclesWaiting'].min():.0f}",
                "-"
            ]
        }
        
        # Calculate improvement
        ml_throughput = ml_episode_df['TotalVehicles'].mean()
        static_throughput = static_episode_df['TotalVehicles'].mean()
        improvement = ((ml_throughput - static_throughput) / static_throughput) * 100
        comparison_data['ML Agent'][-1] = f"{improvement:.1f}%"
        comparison_data['Fixed-Time'][-1] = "Baseline (0%)"
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n=== ML vs STATIC COMPARISON ===")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv('ml_vs_static_comparison.csv', index=False)
        print("\nComparison table saved to: ml_vs_static_comparison.csv")

if __name__ == "__main__":
    analyze_comparison_results()
