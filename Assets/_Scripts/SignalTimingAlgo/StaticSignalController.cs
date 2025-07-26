using UnityEngine;
using Simulator.TrafficSignal;
using Simulator.ScriptableObject;

[RequireComponent(typeof(TrafficLightSetup), typeof(IntersectionDataCalculator))]
public class StaticSignalController : MonoBehaviour {
    
    [Header("CSV Logging")]
    public bool enableLogging = true;
    
    [Header("Interval Logging Configuration")]
    [SerializeField] private float loggingInterval = 10f;  // NEW: Adjustable interval in seconds
    
    [Header("Static Timing Configuration")]
    public StaticSignalTimingSO staticSignalAlgorithm;
    
    private CsvLogger episodeLogger;
    private CsvLogger rewardLogger;
    private CsvLogger intervalLogger;  // NEW: Interval logger
    
    private TrafficLightSetup trafficLightSetup;
    private IntersectionDataCalculator intersectionDataCalculator;
    
    private int episodeCounter = 0;
    private float episodeStartTime;
    private float simulationStartTime;
    private float lastIntervalLogTime = 0f;  // NEW: Track last interval log time
    
    void Start() {
        trafficLightSetup = GetComponent<TrafficLightSetup>();
        intersectionDataCalculator = GetComponent<IntersectionDataCalculator>();
        
        // Create default static algorithm if none assigned
        if (staticSignalAlgorithm == null) {
            staticSignalAlgorithm = ScriptableObject.CreateInstance<StaticSignalTimingSO>();
        }
        
        if (enableLogging) {
            InitializeLoggers();
        }
        
        simulationStartTime = Time.time;
        StartEpisode();
    }
    
    void InitializeLoggers() {
        episodeLogger = new CsvLogger("static_episode_results.csv", 
            "Episode", 
            "TotalVehicles", 
            "VehiclesWaiting", 
            "EpisodeDuration", 
            "AverageWaitTime",
            "Throughput",
            "CurrentPhase",
            "PhaseGreenTime", 
            "FuelConsumed");
            
        rewardLogger = new CsvLogger("static_reward_progress.csv",
            "Step",
            "Episode", 
            "TotalVehicles",
            "VehiclesWaiting",
            "SimulationTime");

        // NEW: Initialize interval logger
        intervalLogger = new CsvLogger("static_interval_data.csv",
            "SimulationTime",
            "Episode", 
            "TotalVehicles",
            "VehiclesWaiting",
            "QueueLength",
            "AverageWaitTime",
            "Throughput",
            "CurrentPhase",
            "PhaseGreenTime",
            "PhaseDuration",
            "FuelConsumed",
            "VehiclesDeparted",
            "TrafficDensity");
    }
    
    void StartEpisode() {
        episodeCounter++;
        episodeStartTime = Time.time;
        lastIntervalLogTime = 0f;  // NEW: Reset interval log time
        
        // Start logging coroutine
        if (enableLogging) {
            StartCoroutine(LogMetrics());
        }
    }
    
    System.Collections.IEnumerator LogMetrics() {
        int stepCounter = 0;
        
        while (true) {
            yield return new WaitForSeconds(1f); // Log every second
            stepCounter++;
            
            // NEW: Check if it's time for interval logging
            float currentSimulationTime = Time.time - simulationStartTime;
            if (currentSimulationTime >= lastIntervalLogTime + loggingInterval) {
                LogIntervalData(currentSimulationTime);
                lastIntervalLogTime = currentSimulationTime;
            }
            
            // Log step-level data
            rewardLogger?.LogRow(
                stepCounter,
                episodeCounter,
                intersectionDataCalculator.TotalNumberOfVehicles,
                intersectionDataCalculator.TotalNumberOfVehiclesWaitingInIntersection,
                currentSimulationTime
            );
            
            // Log episode data every 30 seconds (simulate episodes)
            if (stepCounter % 30 == 0) {
                LogEpisodeData();
            }
        }
    }

    // NEW: Method to log interval data
    void LogIntervalData(float currentSimulationTime) {
        float avgWaitTime = CalculateAverageWaitTime();
        float throughput = CalculateThroughput();
        float queueLength = CalculateCurrentQueueLength();
        float phaseDuration = CalculateCurrentPhaseDuration();
        int vehiclesDeparted = CalculateVehiclesDeparted();
        float trafficDensity = CalculateTrafficDensity();
        
        intervalLogger?.LogRow(
            currentSimulationTime,
            episodeCounter,
            intersectionDataCalculator.TotalNumberOfVehicles,
            intersectionDataCalculator.TotalNumberOfVehiclesWaitingInIntersection,
            queueLength,
            avgWaitTime,
            throughput,
            trafficLightSetup.CurrentPhaseIndex,
            trafficLightSetup.Phases[trafficLightSetup.CurrentPhaseIndex].greenLightTime,
            phaseDuration,
            intersectionDataCalculator.totalFuelConsumed,
            vehiclesDeparted,
            trafficDensity
        );
        
        Debug.Log($"Interval data logged at simulation time: {currentSimulationTime:F1}s");
    }

    void LogEpisodeData() {
        float avgWaitTime = CalculateAverageWaitTime();
        float throughput = CalculateThroughput();
        float fuel = intersectionDataCalculator.totalFuelConsumed;

        episodeLogger?.LogRow(
            episodeCounter,
            intersectionDataCalculator.TotalNumberOfVehicles,
            intersectionDataCalculator.TotalNumberOfVehiclesWaitingInIntersection,
            Time.time - episodeStartTime,
            avgWaitTime,
            throughput,
            trafficLightSetup.CurrentPhaseIndex,
            trafficLightSetup.Phases[trafficLightSetup.CurrentPhaseIndex].greenLightTime,
            fuel
        );
        
        episodeCounter += 1;
        episodeStartTime = Time.time;
        intersectionDataCalculator.totalFuelConsumed = 0f;
    }
    
    float CalculateAverageWaitTime() {
        // Calculate average wait time from intersection data
        if (intersectionDataCalculator.vehiclesWaitingAtLeg == null) return 0f;
        
        float totalWaitTime = 0f;
        int totalVehicles = 0;
        
        foreach (var leg in intersectionDataCalculator.vehiclesWaitingAtLeg) {
            if (leg != null) {
                foreach (var vehicle in leg.Values) {
                    totalWaitTime += vehicle;
                    totalVehicles++;
                }
            }
        }
        
        return totalVehicles > 0 ? totalWaitTime / totalVehicles : 0f;
    }

    // NEW: Calculate current throughput
    float CalculateThroughput() {
        float elapsedTime = Time.time - episodeStartTime;
        return elapsedTime > 0 ? intersectionDataCalculator.TotalNumberOfVehicles / elapsedTime : 0f;
    }

    // NEW: Calculate current queue length across all legs
    float CalculateCurrentQueueLength() {
        return intersectionDataCalculator.TotalNumberOfVehiclesWaitingInIntersection;
    }

    // NEW: Calculate how long current phase has been active
    float CalculateCurrentPhaseDuration() {
        // You might need to track this in TrafficLightSetup or add phase start time tracking
        // For now, returning the configured green light time
        return trafficLightSetup.Phases[trafficLightSetup.CurrentPhaseIndex].greenLightTime;
    }

    // NEW: Calculate vehicles that have departed (you may need to implement this)
    int CalculateVehiclesDeparted() {
        // This would require tracking vehicles that have left the intersection
        // Placeholder implementation - you may need to add this tracking
        return intersectionDataCalculator.TotalNumberOfVehicles - intersectionDataCalculator.TotalNumberOfVehiclesWaitingInIntersection;
    }

    // NEW: Calculate traffic density
    float CalculateTrafficDensity() {
        // Simple density calculation - vehicles per unit area or similar metric
        // You can customize this based on your intersection size/area
        return intersectionDataCalculator.TotalNumberOfVehicles / 1f; // Placeholder denominator
    }

    // NEW: Public method to change logging interval at runtime
    public void SetLoggingInterval(float newInterval) {
        loggingInterval = newInterval;
        Debug.Log($"Logging interval changed to {newInterval} seconds");
    }
    
    void OnApplicationQuit() {
        SaveAllData();
    }
    
    void OnDestroy() {
        SaveAllData();
    }
    
    void SaveAllData() {
        episodeLogger?.SaveToFile();
        rewardLogger?.SaveToFile();
        intervalLogger?.SaveToFile();  // NEW: Save interval data
    }
    
    [ContextMenu("Save Static CSV Data")]
    public void ManualSave() {
        SaveAllData();
    }
}
