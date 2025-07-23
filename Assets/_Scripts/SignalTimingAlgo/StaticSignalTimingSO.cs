using UnityEngine;
using Simulator.TrafficSignal;

namespace Simulator.ScriptableObject {
    [CreateAssetMenu(menuName = "ScriptableObjects/MLAlgorithm/StaticSignalTiming", fileName = "DefaultStaticSignalTiming", order = 2)]
    public class StaticSignalTimingSO : UnityEngine.ScriptableObject {
        
        [Header("Fixed Timing Settings")]
        public float[] phaseTimings = { 30f, 30f, 30f, 30f }; // Default timings for each phase
        public bool useFixedCycleTime = true;
        public float fixedCycleTime = 120f; // Total cycle time
        
        /// <summary>
        /// Gets the next phase using fixed timing
        /// </summary>
        /// <param name="intersectionDataCalculator">Intersection data for metrics</param>
        /// <param name="currentPhaseIndex">Current active phase</param>
        /// <returns>Tuple: (next phase index, green light duration)</returns>
        public (int, float) GetNextPhase(IntersectionDataCalculator intersectionDataCalculator, int currentPhaseIndex) {
            // For fixed timing, we just cycle through phases with predetermined durations
            int nextPhaseIndex = -1; // -1 means go to next phase in sequence
            
            float nextPhaseDuration;
            if (useFixedCycleTime) {
                // Divide total cycle time equally among all phases
                nextPhaseDuration = fixedCycleTime / phaseTimings.Length;
            } else {
                // Use predefined phase timings
                int nextIndex = (currentPhaseIndex + 1) % phaseTimings.Length;
                nextPhaseDuration = phaseTimings[nextIndex];
            }
            
            return (nextPhaseIndex, nextPhaseDuration);
        }
    }
}
