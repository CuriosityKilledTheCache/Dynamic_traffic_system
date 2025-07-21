using System.IO;
using System.Text;
using UnityEngine;

public class CsvLogger {
    private string filePath;
    private StringBuilder csvContent;
    private bool headerWritten = false;

    public CsvLogger(string fileName, params string[] columnNames) {
        filePath = Path.Combine(Application.persistentDataPath, fileName);
        csvContent = new StringBuilder();

        // Write headers
        csvContent.AppendLine(string.Join(",", columnNames));
        headerWritten = true;
    }

    public void LogRow(params object[] values) {
        string[] stringValues = new string[values.Length];
        for (int i = 0; i < values.Length; i++) {
            stringValues[i] = values[i].ToString();
        }
        csvContent.AppendLine(string.Join(",", stringValues));
    }

    public void SaveToFile() {
        File.WriteAllText(filePath, csvContent.ToString());
        Debug.Log($"CSV saved to: {filePath}");
    }

    public void Clear() {
        csvContent.Clear();
        if (headerWritten) {
            // Re-add headers if they were there before
            // This is handled in constructor, so we just clear content
        }
    }
}
