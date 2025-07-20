using Simulator.TrafficSignal;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class TrafficSignalMlAgent : Agent
{
    [Header("Traffic Signal Components")]
    [SerializeField] private TrafficSignal targetSignal;
    [SerializeField] private Transform intersectionCenter;
    [SerializeField] private float detectionRadius = 100f;
    
    [Header("Detection Points for Queue Length")]
    [SerializeField] private Transform[] detectionPoints; // 4 directions: North, East, South, West
    
    [Header("Performance Tracking")]
    [SerializeField] private bool enablePerformanceLogging = true;
    
    // State observation storage
    private float[] currentObservations = new float[8];
    
    // Performance tracking
    private float episodeStartTime;
    private float totalRewardThisEpisode = 0f;
    
    void Start()
    {
        if (targetSignal == null)
            targetSignal = GetComponent<TrafficSignal>();
        
        if (intersectionCenter == null)
            intersectionCenter = transform;
            
        // Initialize detection points if not assigned
        if (detectionPoints == null || detectionPoints.Length != 4)
        {
            CreateDetectionPoints();
        }
        
        episodeStartTime = Time.time;
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Observation space: 8 values total
        
        // 1-4: Queue lengths from 4 directions (normalized 0-1)
        for (int i = 0; i < 4; i++)
        {
            float queueLength = GetQueueLength(i);
            float normalizedQueue = Mathf.Clamp01(queueLength / 20f); // Max 20 vehicles
            sensor.AddObservation(normalizedQueue);
            currentObservations[i] = normalizedQueue;
        }
        
        // 5: Current signal phase (0-3, normalized)
        float currentPhase = targetSignal != null ? targetSignal.CurrentPhase : 0f;
        float normalizedPhase = currentPhase / 3f;
        sensor.AddObservation(normalizedPhase);
        currentObservations[4] = normalizedPhase;
        
        // 6: Remaining time in current phase (0-1)
        float remainingTime = targetSignal != null ? targetSignal.RemainingTime : 0f;
        float maxPhaseTime = targetSignal != null ? targetSignal.MaxPhaseTime : 30f;
        float normalizedTime = Mathf.Clamp01(remainingTime / maxPhaseTime);
        sensor.AddObservation(normalizedTime);
        currentObservations[5] = normalizedTime;
        
        // 7: Time of day factor (0-1, cycles every 300 seconds)
        float timeOfDay = (Time.time % 300f) / 300f;
        sensor.AddObservation(timeOfDay);
        currentObservations[6] = timeOfDay;
        
        // 8: Total vehicles near intersection (normalized)
        float totalVehicles = GetTotalVehiclesNearIntersection();
        float normalizedTotal = Mathf.Clamp01(totalVehicles / 50f); // Max 50 vehicles
        sensor.AddObservation(normalizedTotal);
        currentObservations[7] = normalizedTotal;
        
        if (enablePerformanceLogging)
        {
            LogObservations();
        }
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];
        
        // Action space:
        // 0: Continue current phase (no change)
        // 1-4: Switch to specific phase (0-3)
        
        bool phaseChanged = false;
        
        if (action >= 1 && action <= 4 && targetSignal != null)
        {
            int requestedPhase = action - 1;
            if (requestedPhase != targetSignal.CurrentPhase)
            {
                targetSignal.RequestPhaseChange(requestedPhase);
                phaseChanged = true;
            }
        }
        
        // Calculate reward
        float reward = CalculateReward(phaseChanged);
        AddReward(reward);
        totalRewardThisEpisode += reward;
        
        // Register metrics with QuickMetricsCollector
        if (QuickMetricsCollector.Instance != null)
        {
            QuickMetricsCollector.Instance.RegisterAgentAction(action, reward, GetTotalWaitingTime());
        }
    }
    
    private float CalculateReward(bool phaseChanged)
    {
        float reward = 0f;
        
        // Primary reward: Negative reward for total waiting time
        float totalWaitTime = GetTotalWaitingTime();
        reward -= totalWaitTime * 0.01f;
        
        // Secondary reward: Negative reward for long queues
        float totalQueuePenalty = 0f;
        for (int i = 0; i < 4; i++)
        {
            float queueLength = GetQueueLength(i);
            if (queueLength > 10f)
            {
                totalQueuePenalty += (queueLength - 10f) * 0.005f;
            }
        }
        reward -= totalQueuePenalty;
        
        // Efficiency bonus: Small positive reward for keeping traffic flowing
        float totalVehicles = GetTotalVehiclesNearIntersection();
        if (totalVehicles > 0)
        {
            float flowingVehicles = GetMovingVehiclesCount();
            float flowRatio = flowingVehicles / totalVehicles;
            reward += flowRatio * 0.02f;
        }
        
        // Phase change penalty (to prevent excessive switching)
        if (phaseChanged)
        {
            reward -= 0.01f;
        }
        
        return reward;
    }
    
    private float GetQueueLength(int direction)
    {
        if (detectionPoints == null || direction >= detectionPoints.Length)
            return 0f;
            
        Vector3 detectionPoint = detectionPoints[direction].position;
        return VehicleManager.Instance?.GetVehicleCountInRadius(detectionPoint, 50f) ?? 0f;
    }
    
    private float GetTotalWaitingTime()
    {
        return VehicleManager.Instance?.GetTotalWaitingTimeNearIntersection(intersectionCenter.position, detectionRadius) ?? 0f;
    }
    
    private float GetTotalVehiclesNearIntersection()
    {
        return VehicleManager.Instance?.GetVehicleCountInRadius(intersectionCenter.position, detectionRadius) ?? 0f;
    }
    
    private float GetMovingVehiclesCount()
    {
        return VehicleManager.Instance?.GetMovingVehiclesInRadius(intersectionCenter.position, detectionRadius) ?? 0f;
    }
    
    public override void OnEpisodeBegin()
    {
        episodeStartTime = Time.time;
        totalRewardThisEpisode = 0f;
        
        // Reset traffic signal to initial state
        if (targetSignal != null)
        {
            targetSignal.ResetToInitialPhase();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Simple heuristic: choose phase with longest queue
        var discreteActions = actionsOut.DiscreteActions;
        
        float maxQueue = -1f;
        int bestPhase = 0;
        
        for (int i = 0; i < 4; i++)
        {
            float queueLength = GetQueueLength(i);
            if (queueLength > maxQueue)
            {
                maxQueue = queueLength;
                bestPhase = i + 1; // +1 because action 0 is "continue current"
            }
        }
        
        discreteActions[0] = bestPhase;
    }
    
    private void CreateDetectionPoints()
    {
        detectionPoints = new Transform[4];
        
        string[] directions = {"North", "East", "South", "West"};
        Vector3[] offsets = {
            Vector3.forward * 75f,    // North
            Vector3.right * 75f,      // East
            Vector3.back * 75f,       // South
            Vector3.left * 75f        // West
        };
        
        for (int i = 0; i < 4; i++)
        {
            GameObject detectionPoint = new GameObject($"DetectionPoint_{directions[i]}");
            detectionPoint.transform.SetParent(transform);
            detectionPoint.transform.position = intersectionCenter.position + offsets[i];
            detectionPoints[i] = detectionPoint.transform;
        }
    }
    
    private void LogObservations()
    {
        if (Time.frameCount % 300 == 0) // Log every 5 seconds at 60fps
        {
            Debug.Log($"[ML Agent] Queues: N={currentObservations[0]:F2}, E={currentObservations[1]:F2}, " +
                     $"S={currentObservations[2]:F2}, W={currentObservations[3]:F2}, " +
                     $"Phase={currentObservations[4]:F2}, Reward/Episode={totalRewardThisEpisode:F2}");
        }
    }
}
