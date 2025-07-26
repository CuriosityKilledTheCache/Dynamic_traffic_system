using Simulator.TrafficSignal;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Simulator.Manager;

namespace Simulator.SignalTiming {
    [System.Serializable]
    public class ML_DATA {
        public int OFSET;
        public int NUM_OF_LEGS;
        public int NUM_OF_VEHICLES_PER_LEG;
        public int NUM_OF_OBSERVATIONS_PER_VEHICLE;
        public float MINIMUM_GREEN_LIGHT_OFSET;
        public float MAXIMUM_GREEN_LIGHT_OFSET;
        public float[] observations;
        public float rewards;
    }

    [RequireComponent(typeof(TrafficLightSetup))]
    public class TrafficSignalMlAgent : Agent {

        public ML_DATA Ml_data;

        // CSV Loggers
        private CsvLogger episodeLogger;
        private CsvLogger rewardLogger;
        private CsvLogger intervalLogger;  // NEW: Interval logger
        
        private int episodeCounter = 0;
        private float episodeStartTime;

        // NEW: Interval logging variables
        [SerializeField] private float loggingInterval = 10f;  // Adjustable in inspector
        private float lastLogTime = 0f;
        private float simulationStartTime;

        private float action;
        private TrafficLightSetup trafficLightSetup;
        private float greenLightTime;
        private Phase[] phases;

        public override void Initialize() {
            Academy.Instance.AutomaticSteppingEnabled = false;
            base.Awake();
            trafficLightSetup = GetComponent<TrafficLightSetup>();
            phases = trafficLightSetup.Phases;
            Ml_data.observations = new float[Ml_data.OFSET + (Ml_data.NUM_OF_LEGS * Ml_data.NUM_OF_VEHICLES_PER_LEG * Ml_data.NUM_OF_OBSERVATIONS_PER_VEHICLE)];

            // INITIALIZE LOGGERS
            episodeLogger = new CsvLogger("episode_results.csv",
                "Episode",
                "TotalVehicles",
                "VehiclesWaiting",
                "EpisodeDuration",
                "CumulativeReward",
                "CurrentReward",
                "CurrentPhase",
                "GreenLightTime", 
                "FuelConsumed");

            rewardLogger = new CsvLogger("reward_progress.csv",
                "Step",
                "Episode",
                "Reward",
                "CumulativeReward");

            // NEW: Initialize interval logger
            intervalLogger = new CsvLogger("interval_data.csv",
                "SimulationTime",
                "Episode",
                "Step",
                "TotalVehicles",
                "VehiclesWaiting",
                "QueueLength",
                "CumulativeReward",
                "CurrentReward",
                "CurrentPhase",
                "GreenLightTime",
                "FuelConsumed",
                "AverageWaitTime");

            episodeStartTime = Time.time;
            simulationStartTime = Time.time;  // NEW: Track simulation start time
            lastLogTime = 0f;  // NEW: Initialize last log time
        }

        public override void OnEpisodeBegin() {
            Reset();
            episodeCounter++;
            episodeStartTime = Time.time;
            GameManager.Instance.TotalFuelUsed = 0f;
        }

        // NEW: Method to log interval data
        private void LogIntervalData() {
            float currentSimulationTime = Time.time - simulationStartTime;
            
            if (currentSimulationTime >= lastLogTime + loggingInterval) {
                var intersectionData = trafficLightSetup?.GetComponent<IntersectionDataCalculator>();
                
                if (intersectionData != null) {
                    // Calculate additional metrics
                    float queueLength = CalculateCurrentQueueLength();
                    float averageWaitTime = CalculateAverageWaitTime();
                    
                    intervalLogger.LogRow(
                        currentSimulationTime,
                        episodeCounter,
                        StepCount,
                        intersectionData.TotalNumberOfVehicles,
                        intersectionData.TotalNumberOfVehiclesWaitingInIntersection,
                        queueLength,
                        GetCumulativeReward(),
                        Ml_data.rewards,
                        trafficLightSetup.CurrentPhaseIndex,
                        greenLightTime,
                        GameManager.Instance.TotalFuelUsed,
                        averageWaitTime
                    );
                }
                
                lastLogTime = currentSimulationTime;
                Debug.Log($"Interval data logged at simulation time: {currentSimulationTime:F1}s");
            }
        }

        // NEW: Calculate current queue length across all legs
        private float CalculateCurrentQueueLength() {
            // You'll need to implement this based on your traffic system
            // This is a placeholder - adapt to your actual queue calculation
            var intersectionData = trafficLightSetup?.GetComponent<IntersectionDataCalculator>();
            return intersectionData?.TotalNumberOfVehiclesWaitingInIntersection ?? 0f;
        }

        // NEW: Calculate average wait time
        private float CalculateAverageWaitTime() {
            // You'll need to implement this based on your traffic system
            // This is a placeholder - adapt to your actual wait time calculation
            return 0f; // Replace with actual calculation
        }

        // NEW: Update method to handle interval logging
        private void Update() {
            // Log data at specified intervals
            LogIntervalData();
        }

        // NEW: Public method to change logging interval at runtime
        public void SetLoggingInterval(float newInterval) {
            loggingInterval = newInterval;
            Debug.Log($"Logging interval changed to {newInterval} seconds");
        }

        private float ChangeToNextPhaseWithTimeInterpolate(float time) {
            int index = (trafficLightSetup.CurrentPhaseIndex + 1) % phases.Length;
            return Mathf.FloorToInt(Mathf.Lerp(phases[index].greenLightTime - Ml_data.MINIMUM_GREEN_LIGHT_OFSET, phases[index].greenLightTime + Ml_data.MAXIMUM_GREEN_LIGHT_OFSET, (time + 1) / 2));
        }

        public (int, float) GenerateAction() {
            // Log step-level reward data
            rewardLogger.LogRow(
                StepCount,
                episodeCounter,
                Ml_data.rewards,
                GetCumulativeReward()
            );

            AddReward(Ml_data.rewards);
            float fuel = GameManager.Instance.TotalFuelUsed;

            // LOG EPISODE DATA BEFORE ENDING
            if (trafficLightSetup != null) {
                var intersectionData = trafficLightSetup.GetComponent<IntersectionDataCalculator>();
                if (intersectionData != null) {
                    episodeLogger.LogRow(
                        episodeCounter,
                        intersectionData.TotalNumberOfVehicles,
                        intersectionData.TotalNumberOfVehiclesWaitingInIntersection,
                        Time.time - episodeStartTime,
                        GetCumulativeReward(),
                        Ml_data.rewards,
                        trafficLightSetup.CurrentPhaseIndex,
                        greenLightTime,
                        fuel
                    );
                }
            }

            EndEpisode();
            Academy.Instance.EnvironmentStep();
            RequestDecision();

            greenLightTime = ChangeToNextPhaseWithTimeInterpolate(action);
            return (-1, greenLightTime);
        }

        public override void CollectObservations(VectorSensor sensor) {
            sensor.AddObservation(Ml_data.observations);
        }

        public override void OnActionReceived(ActionBuffers actions) {
            action = actions.ContinuousActions[0];
        }

        public void Reset() {
            // Reset logic here
        }

        // Save data when application quits or gets destroyed
        private void OnApplicationQuit() {
            SaveAllData();
        }

        private void OnDestroy() {
            SaveAllData();
        }

        private void SaveAllData() {
            episodeLogger?.SaveToFile();
            rewardLogger?.SaveToFile();
            intervalLogger?.SaveToFile();  // NEW: Save interval data
        }

        // Manual save method
        [ContextMenu("Save CSV Data")]
        public void ManualSave() {
            SaveAllData();
        }
    }
}
