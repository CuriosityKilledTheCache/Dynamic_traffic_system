using UnityEngine;
using Simulator.TrafficSignal;
using Simulator.ScriptableObject;

[RequireComponent(typeof(TrafficLightSetup), typeof(IntersectionDataCalculator))]
public class StaticSignalController : MonoBehaviour {
    
    [Header("CSV Logging")]
    public bool enableLogging = true;
    
    [Header("Static Timing Configuration")]
    public StaticSignalTimingSO staticSignalAlgorithm;
    
    private CsvLogger episodeLogger;
    private CsvLogger rewardLogger;
    private TrafficLightSetup trafficLightSetup;
    private IntersectionDataCalculator intersectionDataCalculator;
    
    private int episodeCounter = 0;
    private float episodeStartTime;
    private float simulationStartTime;
    
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
    }
    
    void StartEpisode() {
        episodeCounter++;
        episodeStartTime = Time.time;
        
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
            
            // Log step-level data
            rewardLogger?.LogRow(
                stepCounter,
                episodeCounter,
                intersectionDataCalculator.TotalNumberOfVehicles,
                intersectionDataCalculator.TotalNumberOfVehiclesWaitingInIntersection,
                Time.time - simulationStartTime
            );
            
            // Log episode data every 30 seconds (simulate episodes)
            if (stepCounter % 30 == 0) {
                LogEpisodeData();
            }
        }
    }

    void LogEpisodeData() {
        float avgWaitTime = CalculateAverageWaitTime();
        float throughput = intersectionDataCalculator.TotalNumberOfVehicles / (Time.time - episodeStartTime);
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
        
        episodeCounter += 1;          // NEW: advance episode index
        episodeStartTime = Time.time; // optional: reset episode clock
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
    
    void OnApplicationQuit() {
        SaveAllData();
    }
    
    void OnDestroy() {
        SaveAllData();
    }
    
    void SaveAllData() {
        episodeLogger?.SaveToFile();
        rewardLogger?.SaveToFile();
    }
    
    [ContextMenu("Save Static CSV Data")]
    public void ManualSave() {
        SaveAllData();
    }
}
