using UnityEngine;

public class SignalSettingsProvider : MonoBehaviour
{
    // Drag your SignalSettingsManager asset here in the Inspector
    public SignalSettingsManager settingsManager;

    private void Awake()
    {
        if (settingsManager == null)
            Debug.LogError("Please assign the SignalSettingsManager asset!", this);
        SignalSettingsLocator.Provider = settingsManager;
    }
}
