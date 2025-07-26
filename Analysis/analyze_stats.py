import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TrafficSignalComparison:
    def __init__(self, data_directory="./"):
        """
        Initialize the comparison class
        
        Args:
            data_directory: Directory containing the CSV files
        """
        self.data_dir = Path(data_directory)
        self.ml_data = {}
        self.static_data = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        # sns.set_palette("husl")
        sns.set_palette(["#2E86AB", "#F24236", "#A23B72", "#F18F01", "#C73E1D"])
        
    def load_data(self):
        """Load all CSV files for both ML and static approaches"""
        try:
            # Load ML data
            self.ml_data['episodes'] = pd.read_csv(self.data_dir / 'episode_results.csv')
            self.ml_data['rewards'] = pd.read_csv(self.data_dir / 'reward_progress.csv')
            self.ml_data['intervals'] = pd.read_csv(self.data_dir / 'interval_data.csv')
            
            # Load Static data
            self.static_data['episodes'] = pd.read_csv(self.data_dir / 'static_episode_results.csv')
            self.static_data['rewards'] = pd.read_csv(self.data_dir / 'static_reward_progress.csv')
            self.static_data['intervals'] = pd.read_csv(self.data_dir / 'static_interval_data.csv')
            
            print("✅ All CSV files loaded successfully!")
            self.print_data_summary()
            
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            print("Make sure all CSV files are in the specified directory")
            
    def print_data_summary(self):
        """Print summary of loaded data"""
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        for approach in ['ml', 'static']:
            data = self.ml_data if approach == 'ml' else self.static_data
            print(f"\n{approach.upper()} Data:")
            
            for file_type, df in data.items():
                print(f"  {file_type}: {len(df)} rows, {len(df.columns)} columns")
                
    def compare_episode_performance(self):
        """Compare episode-level performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Episode Performance Comparison: ML vs Static', fontsize=16, fontweight='bold')
        
        ml_episodes = self.ml_data['episodes']
        static_episodes = self.static_data['episodes']
        
        metrics = [
            ('TotalVehicles', 'Total Vehicles'),
            ('VehiclesWaiting', 'Vehicles Waiting'),
            ('EpisodeDuration', 'Episode Duration (s)'),
            ('FuelConsumed', 'Fuel Consumed'),
            ('CumulativeReward', 'Cumulative Reward'),
            ('GreenLightTime', 'Green Light Time (s)')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Check if metric exists in both datasets
            if metric in ml_episodes.columns and metric in static_episodes.columns:
                # Box plot comparison
                data_to_plot = [ml_episodes[metric].dropna(), static_episodes[metric].dropna()]
                labels = ['ML Agent', 'Static Controller']
                
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                # bp['boxes'][0].set_facecolor('lightblue')
                # bp['boxes'][1].set_facecolor('lightcoral')
                bp['boxes'][0].set_facecolor('#2E86AB')  # Dark blue for ML Agent
                bp['boxes'][1].set_facecolor('#F24236')  # Bright red for Static Controller

                
                ax.set_title(f'{title}')
                ax.grid(True, alpha=0.3)
                
                # Add mean values as text
                ml_mean = ml_episodes[metric].mean()
                static_mean = static_episodes[metric].mean()
                ax.text(0.02, 0.98, f'ML Mean: {ml_mean:.2f}\nStatic Mean: {static_mean:.2f}', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'{metric}\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title} - Data Not Available')
        
        plt.tight_layout()
        plt.savefig('episode_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def compare_interval_data(self):
        """Compare interval-based performance over time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Interval Data Comparison: ML vs Static Over Time', fontsize=16, fontweight='bold')
        
        ml_intervals = self.ml_data['intervals']
        static_intervals = self.static_data['intervals']
        
        # Metrics to compare over time
        time_metrics = [
            ('TotalVehicles', 'Total Vehicles Over Time'),
            ('VehiclesWaiting', 'Vehicles Waiting Over Time'),
            ('FuelConsumed', 'Fuel Consumption Over Time'),
            ('QueueLength', 'Queue Length Over Time')
        ]
        
        for i, (metric, title) in enumerate(time_metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if metric in ml_intervals.columns and metric in static_intervals.columns:
                # Plot time series
                # ax.plot(ml_intervals['SimulationTime'], ml_intervals[metric], 
                #        label='ML Agent', linewidth=2, alpha=0.8)
                # ax.plot(static_intervals['SimulationTime'], static_intervals[metric], 
                #        label='Static Controller', linewidth=2, alpha=0.8)
                ax.plot(ml_intervals['SimulationTime'], ml_intervals[metric], 
                        label='ML Agent', linewidth=2, alpha=0.8, color='#2E86AB')  # Dark blue
                ax.plot(static_intervals['SimulationTime'], static_intervals[metric], 
                        label='Static Controller', linewidth=2, alpha=0.8, color='#F24236')  # Bright red

                
                ax.set_xlabel('Simulation Time (s)')
                ax.set_ylabel(metric)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{metric}\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title} - Data Not Available')
        
        plt.tight_layout()
        plt.savefig('interval_data_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def statistical_comparison(self):
        """Perform statistical comparison between ML and Static approaches"""
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        ml_episodes = self.ml_data['episodes']
        static_episodes = self.static_data['episodes']
        
        # Compare key metrics
        comparison_metrics = ['TotalVehicles', 'VehiclesWaiting', 'FuelConsumed', 'EpisodeDuration']
        
        results = []
        
        for metric in comparison_metrics:
            if metric in ml_episodes.columns and metric in static_episodes.columns:
                ml_values = ml_episodes[metric].dropna()
                static_values = static_episodes[metric].dropna()
                
                # Calculate statistics
                ml_stats = {
                    'mean': ml_values.mean(),
                    'std': ml_values.std(),
                    'median': ml_values.median(),
                    'min': ml_values.min(),
                    'max': ml_values.max()
                }
                
                static_stats = {
                    'mean': static_values.mean(),
                    'std': static_values.std(),
                    'median': static_values.median(),
                    'min': static_values.min(),
                    'max': static_values.max()
                }
                
                # Calculate improvement percentage
                improvement = ((static_stats['mean'] - ml_stats['mean']) / static_stats['mean']) * 100
                
                results.append({
                    'Metric': metric,
                    'ML_Mean': ml_stats['mean'],
                    'Static_Mean': static_stats['mean'],
                    'ML_Std': ml_stats['std'],
                    'Static_Std': static_stats['std'],
                    'Improvement_%': improvement
                })
                
                print(f"\n{metric}:")
                print(f"  ML Agent    - Mean: {ml_stats['mean']:.2f}, Std: {ml_stats['std']:.2f}")
                print(f"  Static      - Mean: {static_stats['mean']:.2f}, Std: {static_stats['std']:.2f}")
                print(f"  Improvement: {improvement:.2f}% {'(ML better)' if improvement > 0 else '(Static better)'}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print("="*60)
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
        # Save summary to CSV
        summary_df.to_csv('performance_comparison_summary.csv', index=False)
        print(f"\n📊 Summary saved to 'performance_comparison_summary.csv'")
        
        return summary_df
        
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*70)
        
        ml_episodes = self.ml_data['episodes']
        static_episodes = self.static_data['episodes']
        
        # Overall performance metrics
        metrics_to_analyze = ['TotalVehicles', 'VehiclesWaiting', 'FuelConsumed', 'EpisodeDuration']
        
        ml_better_count = 0
        static_better_count = 0
        
        for metric in metrics_to_analyze:
            if metric in ml_episodes.columns and metric in static_episodes.columns:
                ml_mean = ml_episodes[metric].mean()
                static_mean = static_episodes[metric].mean()
                
                # For most metrics, lower is better (except TotalVehicles which might indicate throughput)
                if metric == 'TotalVehicles':
                    better = 'ML' if ml_mean > static_mean else 'Static'
                else:
                    better = 'ML' if ml_mean < static_mean else 'Static'
                
                if better == 'ML':
                    ml_better_count += 1
                else:
                    static_better_count += 1
        
        print(f"\nOverall Performance Summary:")
        print(f"  Metrics where ML performs better: {ml_better_count}")
        print(f"  Metrics where Static performs better: {static_better_count}")
        
        if ml_better_count > static_better_count:
            print(f"\n🏆 ML Agent shows better overall performance!")
        elif static_better_count > ml_better_count:
            print(f"\n🏆 Static Controller shows better overall performance!")
        else:
            print(f"\n🤝 Both approaches show comparable performance!")
            
        # Learning curve analysis (if reward data available)
        if 'CumulativeReward' in ml_episodes.columns:
            print(f"\nML Learning Analysis:")
            rewards = ml_episodes['CumulativeReward']
            if len(rewards) > 1:
                trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
                print(f"  Reward trend: {'Improving' if trend > 0 else 'Declining'} ({trend:.4f}/episode)")
                print(f"  Final reward: {rewards.iloc[-1]:.2f}")
                print(f"  Best reward: {rewards.max():.2f}")
    
    def create_dashboard(self):
        """Create a comprehensive dashboard with all comparisons"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Traffic Signal Control: ML vs Static Performance Dashboard', 
                    fontsize=20, fontweight='bold')
        
        ml_episodes = self.ml_data['episodes']
        static_episodes = self.static_data['episodes']
        ml_intervals = self.ml_data['intervals']
        static_intervals = self.static_data['intervals']
        
        # 1. Episode comparison - key metrics
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ['TotalVehicles', 'VehiclesWaiting']
        x = np.arange(len(metrics))
        width = 0.35
        
        ml_means = [ml_episodes[m].mean() if m in ml_episodes.columns else 0 for m in metrics]
        static_means = [static_episodes[m].mean() if m in static_episodes.columns else 0 for m in metrics]
        
        # ax1.bar(x - width/2, ml_means, width, label='ML Agent', alpha=0.8)
        # ax1.bar(x + width/2, static_means, width, label='Static Controller', alpha=0.8)
        ax1.bar(x - width/2, ml_means, width, label='ML Agent', alpha=0.8, color='#2E86AB')
        ax1.bar(x + width/2, static_means, width, label='Static Controller', alpha=0.8, color='#F24236')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Average Values')
        ax1.set_title('Episode Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Fuel consumption over time
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'FuelConsumed' in ml_intervals.columns:
            # ax2.plot(ml_intervals['SimulationTime'], ml_intervals['FuelConsumed'], 
            #         label='ML Agent', linewidth=2)
            # ax2.plot(static_intervals['SimulationTime'], static_intervals['FuelConsumed'], 
            #         label='Static Controller', linewidth=2)
            ax2.plot(ml_intervals['SimulationTime'], ml_intervals['FuelConsumed'], 
                    label='ML Agent', linewidth=2, color='#2E86AB')
            ax2.plot(static_intervals['SimulationTime'], static_intervals['FuelConsumed'], 
                    label='Static Controller', linewidth=2, color='#F24236')

            ax2.set_xlabel('Simulation Time (s)')
            ax2.set_ylabel('Fuel Consumed')
            ax2.set_title('Fuel Consumption Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Queue length comparison
        ax3 = fig.add_subplot(gs[1, :2])
        if 'QueueLength' in ml_intervals.columns:
            # ax3.plot(ml_intervals['SimulationTime'], ml_intervals['QueueLength'], 
            #         label='ML Agent', linewidth=2, alpha=0.8)
            # ax3.plot(static_intervals['SimulationTime'], static_intervals['QueueLength'], 
            #         label='Static Controller', linewidth=2, alpha=0.8)
            ax3.plot(ml_intervals['SimulationTime'], ml_intervals['QueueLength'], 
                    label='ML Agent', linewidth=2, alpha=0.8, color='#2E86AB')
            ax3.plot(static_intervals['SimulationTime'], static_intervals['QueueLength'], 
                    label='Static Controller', linewidth=2, alpha=0.8, color='#F24236')
            ax3.set_xlabel('Simulation Time (s)')
            ax3.set_ylabel('Queue Length')
            ax3.set_title('Queue Length Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Reward progression (ML only)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'CumulativeReward' in self.ml_data['rewards'].columns:
            rewards_df = self.ml_data['rewards']
            ax4.plot(rewards_df['Step'], rewards_df['CumulativeReward'], 
                    linewidth=2, color='blue')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Cumulative Reward')
            ax4.set_title('ML Agent Learning Curve')
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics heatmap
        ax5 = fig.add_subplot(gs[2:, :])
        
        # Create performance comparison matrix
        comparison_data = []
        metrics = ['TotalVehicles', 'VehiclesWaiting', 'FuelConsumed', 'EpisodeDuration']
        
        for metric in metrics:
            if metric in ml_episodes.columns and metric in static_episodes.columns:
                ml_val = ml_episodes[metric].mean()
                static_val = static_episodes[metric].mean()
                # Normalize values for better visualization
                comparison_data.append([ml_val, static_val])
        
        if comparison_data:
            comparison_array = np.array(comparison_data)
            # Normalize each row
            for i in range(len(comparison_array)):
                max_val = max(comparison_array[i])
                if max_val > 0:
                    comparison_array[i] = comparison_array[i] / max_val
            
            sns.heatmap(comparison_array, 
                       xticklabels=['ML Agent', 'Static Controller'],
                       yticklabels=[m for m in metrics if m in ml_episodes.columns and m in static_episodes.columns],
                       annot=True, fmt='.3f', cmap='RdYlBu_r',
                       ax=ax5)
            ax5.set_title('Normalized Performance Heatmap')
        
        plt.savefig('traffic_signal_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete comparison analysis"""
        print("🚀 Starting Traffic Signal Comparison Analysis...")
        
        # Load data
        self.load_data()
        
        # Generate all comparisons
        print("\n📊 Generating episode performance comparison...")
        self.compare_episode_performance()
        
        print("\n📈 Generating interval data comparison...")
        self.compare_interval_data()
        
        print("\n🔍 Performing statistical analysis...")
        summary_df = self.statistical_comparison()
        
        print("\n📋 Generating performance report...")
        self.generate_performance_report()
        
        print("\n📊 Creating comprehensive dashboard...")
        self.create_dashboard()
        
        print("\n✅ Analysis complete! Check the generated PNG files and CSV summary.")
        
        return summary_df

# Usage example
if __name__ == "__main__":
    # Initialize the comparison tool
    # Update the path to where your CSV files are located
    comparer = TrafficSignalComparison(data_directory="./")
    
    # Run complete analysis
    summary = comparer.run_complete_analysis()
    
    # Optional: Access individual comparison methods
    # comparer.load_data()
    # comparer.compare_episode_performance()
    # comparer.statistical_comparison()
