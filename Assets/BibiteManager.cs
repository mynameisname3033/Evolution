using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public class BibiteManager : MonoBehaviour
{
    private class BibiteData
    {
        public GameObject obj;
        public Transform transform;
        public Bibite script;
        public Rigidbody2D rb;
        public float[] inputs;
        public float[] weights;
        public float[] biases;
    }

    private List<BibiteData> _bibiteDatas;
    private List<BibiteData> _toRemove;
    private List<BibiteData> _children;

    private List<float> _allInputs;
    private List<float> _allWeights;
    private List<float> _allBiases;

    private int _numInputs;
    private int _numWeights;
    private int _numBiases;

    private int[] weightBaseOffsets;
    private int[] biasBaseOffsets;

    private ComputeBuffer inputsBuffer;
    private ComputeBuffer weightsBuffer;
    private ComputeBuffer biasesBuffer;
    private ComputeBuffer layerSizesBuffer;
    private ComputeBuffer weightBaseOffsetsBuffer;
    private ComputeBuffer biasBaseOffsetsBuffer;
    private ComputeBuffer outputsBuffer;

    private int kernel;

    [Header("Bibite Count")]
    [SerializeField] int numBibites = 20;
    [SerializeField] bool kill5PercentOfBibites = false;

    [Header("Objects")]
    [SerializeField] GameObject _bibitePrefab;
    private NeuralNet _neuralNet;
    [SerializeField] ComputeShader _computeShader;

    [Header("Bibite Properties")]
    [SerializeField] float _maxSpeed = 50f;
    [SerializeField] int _numRays = 5;
    [SerializeField] float _spreadAngle = 90f;
    [SerializeField] int _minChildren = 1;
    [SerializeField] int _maxChildren = 4;

    private void Start()
    {
        _neuralNet = GetComponent<NeuralNet>();

        _bibiteDatas = new List<BibiteData>(numBibites);
        _toRemove = new List<BibiteData>();
        _children = new List<BibiteData>();

        _allInputs = new List<float>();
        _allWeights = new List<float>();
        _allBiases = new List<float>();

        _numInputs = _numRays * 2 + 3;

        for (int i = 0; i < numBibites; i++)
        {
            Vector3 spawnPosition = new(Random.Range(-85, 85), Random.Range(-48, 48), 0);
            GameObject bibiteObj = Instantiate(_bibitePrefab, spawnPosition, Quaternion.Euler(0, 0, Random.Range(0f, 360f)));
            Bibite bibiteScript = bibiteObj.GetComponent<Bibite>();
            Rigidbody2D bibiteRb = bibiteObj.GetComponent<Rigidbody2D>();

            _neuralNet.InitializeParameters(out float[] bibiteWeights, out float[] bibiteBiases);
            _allWeights.AddRange(bibiteWeights);
            _allBiases.AddRange(bibiteBiases);

            bibiteScript._maturityAge = Random.Range(bibiteScript._maturityMinAge, bibiteScript._maturityMaxAge);
            bibiteObj.transform.localScale = Vector3.one * bibiteScript._size;
            bibiteRb.mass = bibiteScript._size;

            _bibiteDatas.Add(new BibiteData
            {
                obj = bibiteObj,
                transform = bibiteObj.transform,
                script = bibiteScript,
                rb = bibiteRb,
                inputs = new float[_numInputs],
                weights = bibiteWeights,
                biases = bibiteBiases
            });
        }

        kernel = _computeShader.FindKernel("Forward");

        _numWeights = _bibiteDatas[0].weights.Length;
        _numBiases = _bibiteDatas[0].biases.Length;

        inputsBuffer = new(_numInputs * 1500, sizeof(float));
        weightsBuffer = new(_numWeights * 1500, sizeof(float));
        biasesBuffer = new(_numBiases * 1500, sizeof(float));
        layerSizesBuffer = new(_neuralNet.layerSizes.Length, sizeof(int));
        weightBaseOffsetsBuffer = new(_neuralNet.layerSizes.Length - 1, sizeof(int));
        biasBaseOffsetsBuffer = new(_neuralNet.layerSizes.Length - 1, sizeof(int));
        outputsBuffer = new(1500, sizeof(int));

        layerSizesBuffer.SetData(_neuralNet.layerSizes);
        weightsBuffer.SetData(_allWeights.ToArray());
        biasesBuffer.SetData(_allBiases.ToArray());

        _computeShader.SetInt("totalLayers", _neuralNet.layerSizes.Length);
        _computeShader.SetInt("numInputs", _numInputs);

        _computeShader.SetBuffer(kernel, "layerSizes", layerSizesBuffer);
        _computeShader.SetBuffer(kernel, "outputs", outputsBuffer);
        _computeShader.SetBuffer(kernel, "weights", weightsBuffer);
        _computeShader.SetBuffer(kernel, "biases", biasesBuffer);

        weightBaseOffsets = new int[_neuralNet.layerSizes.Length - 1];
        biasBaseOffsets = new int[_neuralNet.layerSizes.Length - 1];

        for (int l = 1; l < _neuralNet.layerSizes.Length; l++)
        {
            int prevLayerSize = _neuralNet.layerSizes[l - 1];
            int nextLayerSize = _neuralNet.layerSizes[l];
            weightBaseOffsets[l - 1] = (l - 1) * numBibites * prevLayerSize * nextLayerSize;
            biasBaseOffsets[l - 1] = (l - 1) * numBibites * nextLayerSize;
        }
    }

    private void OnDisable()
    {
        inputsBuffer.Release();
        weightsBuffer.Release();
        biasesBuffer.Release();
        layerSizesBuffer.Release();
        weightBaseOffsetsBuffer.Release();
        biasBaseOffsetsBuffer.Release();
        outputsBuffer.Release();
    }

    private void Update()
    {
        float dt = Time.deltaTime;
        _allInputs.Clear();

        inputsBuffer.SetData(_allInputs.ToArray());
        weightBaseOffsetsBuffer.SetData(weightBaseOffsets);
        biasBaseOffsetsBuffer.SetData(biasBaseOffsets);

        _computeShader.SetBuffer(kernel, "inputs", inputsBuffer);
        _computeShader.SetBuffer(kernel, "weightBaseOffsets", weightBaseOffsetsBuffer);
        _computeShader.SetBuffer(kernel, "biasBaseOffsets", biasBaseOffsetsBuffer);

        _computeShader.SetInt("numBibites", numBibites);

        int threadsPerGroup = 64;
        int threadGroupsX = Mathf.CeilToInt(numBibites / 4f / threadsPerGroup);
        _computeShader.Dispatch(kernel, threadGroupsX, 1, 1);

        int[] outputs = new int[numBibites];
        outputsBuffer.GetData(outputs);

        for (int i = 0; i < numBibites; i++)
        {
            BibiteData bibiteData = _bibiteDatas[i];
            Transform bibiteTransform = bibiteData.transform;
            Vector3 bibiteTransformRight = bibiteTransform.right;
            Bibite bibiteScript = bibiteData.script;
            Rigidbody2D bibiteRb = bibiteData.rb;
            float[] bibiteInputs = bibiteData.inputs;

            bibiteScript._age += dt / 20f;
            if (bibiteScript._age >= bibiteScript._maturityAge && !bibiteScript._matured)
            {
                bibiteScript._matured = true;
                bibiteScript._size += Random.Range(0.1f, 0.35f);
                bibiteRb.mass = bibiteScript._size;
                bibiteTransform.localScale = Vector3.one * bibiteScript._size;
                AddBibites(bibiteData, Random.Range(_minChildren, _maxChildren + 1));
            }

            float movementCost = bibiteRb.linearVelocity.magnitude + Mathf.Abs(bibiteRb.angularVelocity) / 30f;
            float sizeCost = Mathf.Pow(bibiteScript._size, 1.5f);
            float energyDrain = dt * sizeCost * movementCost * 0.2f;
            bibiteScript._energy = Mathf.Max(bibiteScript._energy - energyDrain, 0);

            if (bibiteScript._energy <= 0)
                bibiteScript._health -= dt * 7f * Mathf.Exp(-bibiteScript._health / 200);
            if (bibiteTransform.position.y <= 40)
                bibiteScript._health -= dt * 4f;

            if (bibiteScript._health <= 0 || bibiteScript._age >= bibiteScript._maxAge)
            {
                _toRemove.Add(bibiteData);
                continue;
            }

            int baseIndex = _numRays * 2;
            bibiteInputs[baseIndex] = bibiteTransform.position.x / 55f;
            bibiteInputs[baseIndex + 1] = bibiteTransform.position.y / 55f;
            bibiteInputs[baseIndex + 2] = bibiteTransform.eulerAngles.z / 360f;

            float angleStep = _numRays > 1 ? _spreadAngle / (_numRays - 1) : 0;
            for (int j = 0; j < _numRays; j++)
            {
                float angle = _numRays > 1 ? -_spreadAngle / 2f + j * angleStep : 0;
                Vector2 dir = Quaternion.Euler(0, 0, angle) * bibiteTransformRight;
                RaycastHit2D hit = Physics2D.Raycast(bibiteTransform.position + bibiteTransformRight * bibiteScript._size, dir, bibiteScript._rayLength);
                // Debug.DrawRay(bibiteTransform.position + bibiteTransformRight * bibiteScript._size, dir * bibiteScript._rayLength, Color.red);

                int inputIndex = j * 2;
                if (hit.collider != null)
                {
                    bibiteInputs[inputIndex] = hit.distance / bibiteScript._rayLength;
                    bibiteInputs[inputIndex + 1] = hit.collider.CompareTag("Bibite") ? 1f : 0.5f;
                }
                else
                {
                    bibiteInputs[inputIndex] = 1f;
                    bibiteInputs[inputIndex + 1] = 0f;
                }
            }

            _allInputs.AddRange(bibiteInputs);

            int bestIdx = outputs[i];
            if (bestIdx == 0)
            {
                Vector3 direction = bibiteTransformRight * Mathf.Max(bibiteScript._energy / 75f, 0.1f);
                bibiteRb.AddForce(bibiteScript._speed * direction.normalized);
            }
            else if (bestIdx == 1)
            {
                Vector3 direction = -bibiteRb.linearVelocity * Mathf.Max(bibiteScript._energy / 75f, 0.1f);
                bibiteRb.AddForce(bibiteScript._speed * direction.normalized);
                bibiteScript._health = Mathf.Min(bibiteScript._health + Time.deltaTime * bibiteScript._energy / 100, 100);
                bibiteScript._energy = Mathf.Min(bibiteScript._energy + Time.deltaTime * Mathf.Exp(-bibiteScript._energy / 200), 100);
            }
            else if (bestIdx == 2)
            {
                bibiteRb.AddTorque(-bibiteScript._speed * 4f);
            }
            else
            {
                bibiteRb.AddTorque(bibiteScript._speed * 4f);
            }

            bibiteRb.angularVelocity = Mathf.Clamp(bibiteRb.angularVelocity / 4f, -_maxSpeed, _maxSpeed);
            bibiteRb.linearVelocity = Vector2.ClampMagnitude(bibiteRb.linearVelocity, _maxSpeed);

            bibiteRb.angularVelocity = Mathf.Clamp(bibiteRb.angularVelocity / 4f, -_maxSpeed, _maxSpeed);
            bibiteRb.linearVelocity = Vector2.ClampMagnitude(bibiteRb.linearVelocity, _maxSpeed);
        }

        if (_toRemove.Count > 0) {
            foreach (BibiteData bibiteData in _toRemove)
            {
                int idx = _bibiteDatas.IndexOf(bibiteData);
                _allWeights.RemoveRange(idx, _numWeights);
                _allBiases.RemoveRange(idx, _numBiases);

                weightsBuffer.SetData(_allWeights.ToArray());
                biasesBuffer.SetData(_allBiases.ToArray());
                _computeShader.SetBuffer(kernel, "weights", weightsBuffer);
                _computeShader.SetBuffer(kernel, "biases", biasesBuffer);

                _bibiteDatas.Remove(bibiteData);
                Destroy(bibiteData.obj);
            }

            for (int i = 0; i < weightBaseOffsets.Length; i++)
            {
                weightBaseOffsets[i] /= numBibites;
                weightBaseOffsets[i] *= numBibites + _toRemove.Count;
            }

            for (int i = 0; i < biasBaseOffsets.Length; i++)
            {
                biasBaseOffsets[i] /= numBibites;
                biasBaseOffsets[i] *= numBibites + _toRemove.Count;
            }

            numBibites -= _toRemove.Count;
            _toRemove.Clear();
        }

        if (_children.Count > 0)
        {
            _bibiteDatas.AddRange(_children);

            foreach (BibiteData child in _children)
            {
                _allWeights.AddRange(child.weights);
                _allBiases.AddRange(child.biases);

                weightsBuffer.SetData(_allWeights.ToArray());
                biasesBuffer.SetData(_allBiases.ToArray());
                _computeShader.SetBuffer(kernel, "weights", weightsBuffer);
                _computeShader.SetBuffer(kernel, "biases", biasesBuffer);
            }
        }

        for (int i = 0; i < weightBaseOffsets.Length; i++)
        {
            weightBaseOffsets[i] /= numBibites;
            weightBaseOffsets[i] *= numBibites + _children.Count;
        }

        for (int i = 0; i < biasBaseOffsets.Length; i++)
        {
            biasBaseOffsets[i] /= numBibites;
            biasBaseOffsets[i] *= numBibites + _children.Count;
        }

        numBibites += _children.Count;
        _children.Clear();
    }

    private void OnValidate()
    {
        if (Application.isPlaying)
        {
            if (Time.time > 0.01f)
                numBibites = _bibiteDatas.Count;

            if (kill5PercentOfBibites && numBibites >= 20)
            {
                int numToKill = Mathf.CeilToInt(numBibites / 20f);
                for (int i = 0; i < numToKill; i++)
                {
                    Destroy(_bibiteDatas[0].obj);
                    _bibiteDatas.RemoveAt(0);
                    _allWeights.RemoveRange(0, _numWeights);
                    _allBiases.RemoveRange(0, _numBiases);
                    numBibites--;
                }
            }
        }

        kill5PercentOfBibites = false;
    }

    private void AddBibites(BibiteData parentBibiteData, int numChildren)
    {
        Bibite parentScript = parentBibiteData.script;
        for (int i = 0; i < numChildren; i++)
        {
            Vector3 spawnPos = new(Random.Range(-85f, 85f), Random.Range(-48f, 48f), 0f);
            GameObject bibiteObj = Instantiate(_bibitePrefab, spawnPos, Quaternion.Euler(0, 0, Random.Range(0f, 360f)));
            Bibite bibiteScript = bibiteObj.GetComponent<Bibite>();
            Rigidbody2D bibiteRb = bibiteObj.GetComponent<Rigidbody2D>();

            bibiteScript._generation = parentScript._generation + 1;
            bibiteScript._size = MutateValue(parentScript._size - Random.Range(0.1f, 0.35f), 0.05f, 0.05f);
            bibiteScript._speed = MutateValue(parentScript._speed, 0.3f, 0.1f);
            bibiteScript._rayLength = MutateValue(parentScript._rayLength, 0.2f, 0f);
            bibiteScript._maturityMinAge = MutateValue(parentScript._maturityMinAge, 0.2f, 1f);
            bibiteScript._maturityMaxAge = MutateValue(parentScript._maturityMaxAge, 0.2f, bibiteScript._maturityMinAge + 0.1f);
            bibiteScript._maturityAge = Random.Range(bibiteScript._maturityMinAge, bibiteScript._maturityMaxAge);
            bibiteObj.transform.localScale = Vector3.one * bibiteScript._size;

            _neuralNet.ModifyParameters(parentBibiteData.weights, parentBibiteData.biases, out float[] newWeights, out float[] newBiases);
            bibiteRb.mass = bibiteScript._size;

            _children.Add(new BibiteData
            {
                obj = bibiteObj,
                transform = bibiteObj.transform,
                script = bibiteScript,
                rb = bibiteRb,
                inputs = new float[_numInputs],
                weights = newWeights,
                biases = newBiases
            });
        }
        parentScript._energy -= numChildren * 20;
    }

    private float MutateValue(float baseValue, float range, float minValue = float.NegativeInfinity)
    {
        return Random.Range(0, 2) == 1 ? Mathf.Max(baseValue + Random.Range(-range, range), minValue) : baseValue;
    }
}