using UnityEngine;
using Simulator.TrafficSignal;
using Simulator.ScriptableObject;
using TMPro;

[CreateAssetMenu(menuName = "SignalSettings/Manager",
                 fileName = "GlobalSignalSettingsManager",
                 order = 100)]
public class SignalSettingsManager : ScriptableObject {
    [Header("Which controller should run?")]
    public TrafficSignalAlogrithm algorithmType = TrafficSignalAlogrithm.Static;

    // Timing SOs
    [SerializeField] private StaticSignalTimingSO staticTimingSO;
    [SerializeField] private DynamicSignalTimingSO dynamicTimingSO;
    [SerializeField] private MLSignalTimingOptimizationSO mlTimingSO;
    [SerializeField] private MLPhaseOptimizationSO mlPhaseSO;

    [Header("Phase Definitions")]
    public Phase[] phases;                   // copy from prefab asset
    [Header("Line Renderer")]
    public GameObject lineRendererPrefab;    // copy from prefab
    [Header("UI")]
    public TextMeshPro timingUILabel;        // reference a prefab UI

    // Public getters
    public StaticSignalTimingSO StaticTimingSO => staticTimingSO;
    public DynamicSignalTimingSO DynamicTimingSO => dynamicTimingSO;
    public MLSignalTimingOptimizationSO MlTimingSO => mlTimingSO;
    public MLPhaseOptimizationSO MlPhaseSO => mlPhaseSO;
}
