# TrafficSimulator - Unity ML-Agents Traffic Management System

## Overview

TrafficSimulator is an advanced Unity-based traffic management system that implements **Proximal Policy Optimization (PPO)** reinforcement learning algorithms for dynamic traffic signal timing optimization. The project compares ML-driven traffic control against traditional static timing controllers to demonstrate improved traffic flow efficiency.

## 🚦 Key Features

- **ML-Agents Integration**: PPO-based reinforcement learning for intelligent traffic signal control
- **Static vs Dynamic Comparison**: Built-in comparison between traditional fixed-timing and ML-adaptive controllers
- **Comprehensive Data Logging**: CSV-based logging system for detailed traffic analytics
- **Real-time Visualization**: Unity-based 3D traffic simulation with visual feedback
- **Performance Analysis**: Python-based analysis tools with statistical comparisons and visualizations
- **Configurable Parameters**: Multiple YAML configurations for different training scenarios

## 🏗️ Project Structure

```
TrafficSimulator/
├── Assets/
│   ├── _Scripts/                    # Core C# scripts
│   │   ├── SignalTimingAlgo/        # ML and static signal timing algorithms
│   │   ├── RuntimeData/             # Data calculation and tracking
│   │   ├── Managers/                # Game and object pool managers
│   │   ├── Vehicles/                # Vehicle controller and AI
│   │   └── Road/                    # Road network and graph generation
│   ├── configs/                     # ML-Agents training configurations
│   ├── Scenes/                      # Unity scene files
│   └── Prefabs/                     # Reusable game objects
├── Analysis/                        # Python analysis scripts and results
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🔧 Prerequisites

### Unity Requirements
- **Unity 2022.3 LTS** or later
- **ML-Agents Package 2.0+** (included via Package Manager)

### Python Requirements
- **Python 3.8+** (preferred 3.9)
- **ML-Agents 0.28** (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

### 1. Setup Unity Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CuriosityKilledTheCache/TrafficSimulator.git
   cd TrafficSimulator
   ```

2. **Open in Unity**:
   - Launch Unity Hub
   - Click "Add" and select the cloned directory
   - Open the project (Unity will import packages automatically)

3. **Install ML-Agents Package**:
   - Window → Package Manager
   - Add package from git URL: `com.unity.ml-agents`

### 2. Python Environment Setup

1. **Create virtual environment**:
   ```bash
   python -m venv traffic_env
   source traffic_env/bin/activate  # On Windows: traffic_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Running the Simulation

#### Option A: Training Mode (ML Agent)
1. **Open the Main scene**: `Assets/Scenes/Main.unity`
2. **Configure training**: Edit `Assets/configs/ppo-traffic.yaml` if needed
3. **Start training**:
   ```bash
   mlagents-learn Assets/configs/ppo-traffic.yaml --run-id traffic_experiment_1
   ```
4. **Run Unity**: Press Play in Unity Editor

#### Option B: Comparison Mode (Static vs ML)
1. **Run with inference mode** change Behaviour Parameters of all 4-way-type-1 objects, also drag the trained file in model field
2. **Monitor real-time performance** via Unity's inspector
3. **CSV logs** are automatically generated in `Assets/results/`

## 📊 Key Components

### 1. TrafficSignalMlAgent.cs
- **PPO-based agent** for dynamic signal timing
- **Continuous action space** for green light duration control
- **Multi-metric reward system** (throughput, waiting time, fuel consumption)
- **Real-time CSV logging** for performance analysis

### 2. IntersectionDataCalculator.cs
- **Traffic flow metrics calculation**
- **Vehicle tracking** across intersection legs
- **Wait time and throughput monitoring**
- **Data interface** for ML agent observations

### 3. CsvLogger.cs 
- **Lightweight logging system** for traffic data
- **Episode, interval, and reward logging**
- **Automatic file management** with timestamps

### 4. Analysis Tools
- **Python-based comparison framework**
- **Statistical analysis** (ML vs Static performance)
- **Visualization generation** (charts, dashboards, heatmaps)
- **Performance metrics calculation**

## 🎯 Training Configuration

The project includes several pre-configured training setups:

### Basic PPO Configuration (`fourWay1.yaml`)
```yaml
behaviors:
  TrafficSignalAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 3.0e-4
    max_steps: 500000
    time_horizon: 64
```

### Advanced Configuration (`fourWay1.yaml`)
- **Larger batch sizes** for stable training
- **Extended training duration** (2M steps)
- **Fine-tuned hyperparameters** for traffic scenarios

> fourWay1.yaml was used for training

## 📈 Performance Analysis

### Running Analysis
1. **Generate training data** by running both ML and static controllers
2. **Execute analysis script**:
   ```bash
   cd Analysis/
   python analyze_stats.py
   ```

### Expected Results 
Based on recent analysis runs:
- **ML Agent Total Vehicles**: 660.95 (2.58% improvement over static)
- **Vehicles Waiting Reduction**: 9.07% improvement
- **Statistical significance** demonstrated across multiple metrics

### Generated Outputs
- `episode_performance_comparison.png` - Episode-level performance charts
- `traffic_signal_dashboard.png` - Comprehensive performance dashboard
- `performance_comparison_summary.csv` - Statistical summary
- `queue_length_comparison_*.png` - Queue analysis charts

## 🔬 Research Applications

This simulator is designed for:
- **Traffic engineering research**
- **Reinforcement learning algorithm testing**
- **Urban planning optimization**
- **Comparative analysis** of traffic control strategies
- **Academic studies** on intelligent transportation systems

## 🛠️ Customization

### Adding New Metrics
1. **Extend `IntersectionData` struct** in `IntersectionDataCalculator.cs`
2. **Update CSV headers** in `CsvLogger.cs`
3. **Modify analysis scripts** to include new metrics

### Custom Reward Functions
1. **Edit `MLSignalTimingOptimizationSO.cs`**
2. **Implement new reward calculation logic**
3. **Update observation space** if needed

### New Training Scenarios
1. **Create new YAML config** in `Assets/configs/`
2. **Adjust network architecture** and hyperparameters
3. **Run training** with new configuration

## 📋 Troubleshooting

### Common Issues
1. **ML-Agents not found**: Ensure Python environment is activated
2. **Training not starting**: Check YAML configuration syntax
3. **Performance issues**: Reduce simulation speed or batch size
4. **CSV files empty**: Verify logging initialization in scripts

### Performance Optimization
- **Reduce visual quality** during training
- **Disable unnecessary UI elements**
- **Use smaller buffer sizes** for faster iteration
- **Adjust logging frequency** to reduce I/O overhead

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-algorithm`
3. **Commit changes**: `git commit -am 'Add new traffic algorithm'`
4. **Push to branch**: `git push origin feature/new-algorithm`
5. **Submit Pull Request**

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Unity ML-Agents Team** for the reinforcement learning framework
- **Original repository** by ZennoZenith for the foundational traffic simulation
- **Unity Technologies** for the Universal Render Pipeline and development tools

