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
        //public event Action OnReset;

        // CSV Logger
        private CsvLogger episodeLogger;
        private CsvLogger rewardLogger;
        private int episodeCounter = 0;
        private float episodeStartTime;

        private float action;

        private TrafficLightSetup trafficLightSetup;
        private float greenLightTime;
        private Phase[] phases;

        //protected override void Awake() {
        //    Academy.Instance.AutomaticSteppingEnabled = false;
        //    base.Awake();
        //    trafficLightSetup = GetComponent<TrafficLightSetup>();
        //    phases = trafficLightSetup.Phases;
        //    Ml_data.observations = new float[Ml_data.OFSET + (Ml_data.NUM_OF_LEGS * Ml_data.NUM_OF_VEHICLES_PER_LEG * Ml_data.NUM_OF_OBSERVATIONS_PER_VEHICLE)];
        //}

        public override void Initialize() {
            //base.Initialize();
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

            episodeStartTime = Time.time;
        }


        public override void OnEpisodeBegin() {
            //base.OnEpisodeBegin();
            Reset();

            episodeCounter++;
            episodeStartTime = Time.time;
            GameManager.Instance.TotalFuelUsed = 0f;

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="time"></param>
        /// <returns>
        /// Returns green light time 
        /// </returns>
        private float ChangeToNextPhaseWithTimeInterpolate(float time) {
            int index = (trafficLightSetup.CurrentPhaseIndex + 1) % phases.Length;
            //greenLightTime = Mathf.FloorToInt(Mathf.Lerp(phases[index].minGreenLightTime, phases[index].maxGreenLightTime, (time + 1) / 2));
            return Mathf.FloorToInt(Mathf.Lerp(phases[index].greenLightTime - Ml_data.MINIMUM_GREEN_LIGHT_OFSET, phases[index].greenLightTime + Ml_data.MAXIMUM_GREEN_LIGHT_OFSET, (time + 1) / 2));
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="reward"></param>
        /// <param name="obseve"></param>
        /// <returns> (phaseindex, greenlight time)</returns>
        public (int, float) GenerateAction() {

            // Log step-level reward data
            rewardLogger.LogRow(
                StepCount,
                episodeCounter,
                Ml_data.rewards,
                GetCumulativeReward()
            );

            AddReward(Ml_data.rewards);
            //Debug.Log($"Reward given: {Ml_data.rewards}");
            //Debug.Log(GetCumulativeReward());
            
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

            //print("Decision requested");
            Academy.Instance.EnvironmentStep();
            RequestDecision();
            //print("Decision complete");

            greenLightTime = ChangeToNextPhaseWithTimeInterpolate(action);
            return (-1, greenLightTime);
        }



        public override void CollectObservations(VectorSensor sensor) {
            //base.CollectObservations(sensor);
            sensor.AddObservation(Ml_data.observations);
        }

        public override void OnActionReceived(ActionBuffers actions) {
            //base.OnActionReceived(actions);
            action = actions.ContinuousActions[0];
            //print($"action received: {action}");
        }

        public void Reset() {
            //Ml_data.rewards = 0;
            //Debug.Log("New Episode began");
            //int len = Ml_data.observations.Length;
            //for (int i = 0; i < len; i++) {
            //    Ml_data.observations[i] = -1f;
            //}
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
        }

        // ADD THIS: Manual save method you can call from inspector
        [ContextMenu("Save CSV Data")]
        public void ManualSave() {
            SaveAllData();
        }
    }
}

