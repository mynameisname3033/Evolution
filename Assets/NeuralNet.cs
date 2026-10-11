using UnityEngine;

public class NeuralNet : MonoBehaviour
{
    [SerializeField] int numInputs = 8;
    [SerializeField] int _numHiddenLayers = 2;
    [SerializeField] int _numHiddenNeurons = 4;
    [SerializeField] int _numOutputs = 4;
    [HideInInspector] public int[] layerSizes;

    public void InitializeParameters(out float[] weights, out float[] biases)
    {
        layerSizes = new int[_numHiddenLayers + 2];
        layerSizes[0] = numInputs;

        for (int i = 1; i <= _numHiddenLayers; i++) layerSizes[i] = _numHiddenNeurons;

        layerSizes[_numHiddenLayers + 1] = _numOutputs;

        int totalWeights = 0;
        int totalBiases = 0;

        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            totalWeights += layerSizes[i] * layerSizes[i + 1];
            totalBiases += layerSizes[i + 1];
        }

        weights = new float[totalWeights];
        biases = new float[totalBiases];

        for (int i = 0; i < weights.Length; i++) weights[i] = UnityEngine.Random.Range(-1f, 1f);
        for (int i = 0; i < biases.Length; i++) biases[i] = UnityEngine.Random.Range(-1f, 1f);
    }

    public void ModifyParameters(float[] oldWeights, float[] oldBiases, out float[] newWeights, out float[] newBiases)
    {
        newWeights = (float[])oldWeights.Clone();
        newBiases = (float[])oldBiases.Clone();

        int numWeightsToChange = UnityEngine.Random.Range(1, Mathf.CeilToInt(newWeights.Length / 10f));
        int numBiasesToChange = UnityEngine.Random.Range(1, Mathf.CeilToInt(newBiases.Length / 10f));

        for (int i = 0; i < numWeightsToChange; i++)
        {
            int index = UnityEngine.Random.Range(0, newWeights.Length);
            newWeights[index] += Mathf.Clamp(UnityEngine.Random.Range(-0.005f, 0.005f), -1f, 1f);
        }

        for (int i = 0; i < numBiasesToChange; i++)
        {
            int index = UnityEngine.Random.Range(0, newBiases.Length);
            newBiases[index] += Mathf.Clamp(UnityEngine.Random.Range(-0.005f, 0.005f), -1f, 1f);
        }
    }
}