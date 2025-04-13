using System;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;

public class NeuralNet : MonoBehaviour
{
    [SerializeField] int numInputs = 8;
    [SerializeField] int _numHiddenLayers = 2;
    [SerializeField] int _numHiddenNeurons = 4;
    [SerializeField] int _numOutputs = 4;
    [HideInInspector] public int[] layerSizes;

    private float[] bufferA;
    private float[] bufferB;

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

    public float[] Forward(float[] inputs, float[] weights, float[] biases)
    {
        int maxLayerSize = layerSizes.Max();
        if (bufferA == null || bufferA.Length < maxLayerSize)
        {
            bufferA = new float[maxLayerSize];
            bufferB = new float[maxLayerSize];
        }

        int weightIndex = 0;
        int biasIndex = 0;

        float[] current = bufferA;
        float[] next = bufferB;

        Array.Copy(inputs, current, inputs.Length);

        for (int layer = 0; layer < layerSizes.Length - 1; layer++)
        {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];

            for (int o = 0; o < outputSize; o++)
            {
                float sum = biases[biasIndex++];
                for (int i = 0; i < inputSize; i++)
                {
                    sum += weights[weightIndex++] * current[i];
                }
                next[o] = ReLU(sum); // or use LeakyReLU
            }

            // Swap buffers
            (next, current) = (current, next);
        }

        // Final output is in current
        return current;
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ReLU(float x) => x > 0f ? x : 0f;

    private float LeakyReLU(float x) => x > 0 ? x : 0.01f * x;

    private float[] Softmax(float[] inputs)
    {
        float max = inputs.Max();
        float sum = 0f;
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = Mathf.Exp(inputs[i] - max);
            sum += inputs[i];
        }
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] /= sum;
        }
        return inputs;
    }

    private float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

    private float Tanh(float x) => (float)Math.Tanh(x);
}