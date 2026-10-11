using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Rendering;
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
    private List<BibiteData> _objectPool;

    private List<float> _allInputs;
    private List<float> _allWeights;
    private List<float> _allBiases;
    private int[] neuralNetOutputs;

    private GameObject divider;
    private GameObject[] walls;

    private int _numInputs;
    private int _numWeights;
    private int _numBiases;

    private ComputeBuffer inputsBuffer;
    private ComputeBuffer weightsBuffer;
    private ComputeBuffer biasesBuffer;
    private ComputeBuffer layerSizesBuffer;
    private ComputeBuffer outputsBuffer;

    private int kernel;

    [Header("Bibite Count")]
    [SerializeField] int numBibites = 20;
    [SerializeField] int maxBibites = 1500;
    [SerializeField] bool kill5PercentOfBibites = false;
    [SerializeField] int objectPoolSize = 100;
    [SerializeField] bool add100PooledBibites = false;

    [Header("Objects")]
    [SerializeField] GameObject _bibitePrefab;
    private NeuralNet _neuralNet;
    [SerializeField] ComputeShader _neuralNetShader;
    [SerializeField] ComputeShader _raycastingShader;

    [Header("Bibite Properties")]
    [SerializeField] float _maxSpeed = 50f;
    [SerializeField] int _numRays = 5;
    [SerializeField] float _spreadAngle = 90f;
    [SerializeField] int _minChildren = 1;
    [SerializeField] int _maxChildren = 4;

    private void Start()
    {
        _neuralNet = GetComponent<NeuralNet>();
        divider = GameObject.Find("Divider");
        walls = GameObject.FindGameObjectsWithTag("Wall");

        _bibiteDatas = new List<BibiteData>(numBibites);
        _toRemove = new List<BibiteData>();
        _children = new List<BibiteData>();
        _objectPool = new List<BibiteData>(objectPoolSize);

        _allInputs = new List<float>();
        _allWeights = new List<float>();
        _allBiases = new List<float>();

        _numInputs = _numRays * 2 + 3;
        neuralNetOutputs = new int[numBibites];

        for (int i = 0; i < numBibites; i++)
        {
            Vector3 spawnPosition = new(Random.Range(walls[3].transform.position.x * 1.05f, walls[2].transform.position.x * 1.05f), Random.Range(walls[1].transform.position.y * 1.05f, walls[0].transform.position.y * 1.05f), 0);
            GameObject bibiteObj = Instantiate(_bibitePrefab, spawnPosition, Quaternion.Euler(0, 0, Random.Range(0f, 360f)));
            Bibite bibiteScript = bibiteObj.GetComponent<Bibite>();
            Rigidbody2D bibiteRb = bibiteObj.GetComponent<Rigidbody2D>();

            _neuralNet.InitializeParameters(out float[] bibiteWeights, out float[] bibiteBiases);
            _numWeights = bibiteWeights.Length;
            _numBiases = bibiteBiases.Length;
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

        for (int i = 0; i < objectPoolSize; i++)
        {
            GameObject bibiteObj = Instantiate(_bibitePrefab, Vector3.zero, Quaternion.identity);
            bibiteObj.SetActive(false);

            _objectPool.Add(new BibiteData
            {
                obj = bibiteObj,
                transform = bibiteObj.transform,
                script = bibiteObj.GetComponent<Bibite>(),
                rb = bibiteObj.GetComponent<Rigidbody2D>(),
                inputs = new float[_numInputs]
            });
        }

        kernel = _neuralNetShader.FindKernel("Forward");

        inputsBuffer = new(_numInputs * maxBibites, sizeof(float));
        weightsBuffer = new(_numWeights * maxBibites, sizeof(float));
        biasesBuffer = new(_numBiases * maxBibites, sizeof(float));
        layerSizesBuffer = new(_neuralNet.layerSizes.Length, sizeof(int));
        outputsBuffer = new(maxBibites, sizeof(int));

        layerSizesBuffer.SetData(_neuralNet.layerSizes);
        weightsBuffer.SetData(_allWeights.ToArray());
        biasesBuffer.SetData(_allBiases.ToArray());

        _neuralNetShader.SetInt("totalLayers", _neuralNet.layerSizes.Length);
        _neuralNetShader.SetInt("numInputs", _numInputs);
        _neuralNetShader.SetInt("numWeights", _numWeights);
        _neuralNetShader.SetInt("numBiases", _numBiases);

        _neuralNetShader.SetBuffer(kernel, "layerSizes", layerSizesBuffer);
        _neuralNetShader.SetBuffer(kernel, "outputs", outputsBuffer);
        _neuralNetShader.SetBuffer(kernel, "weights", weightsBuffer);
        _neuralNetShader.SetBuffer(kernel, "biases", biasesBuffer);
    }

    private void OnDisable()
    {
        inputsBuffer.Release();
        weightsBuffer.Release();
        biasesBuffer.Release();
        layerSizesBuffer.Release();
        outputsBuffer.Release();
    }

    private void FixedUpdate()
    {
        if (_toRemove.Count > 0)
        {
            foreach (BibiteData bibiteData in _toRemove)
            {
                int idx = _bibiteDatas.IndexOf(bibiteData);
                if (idx != -1)
                {
                    _allWeights.RemoveRange(idx * _numWeights, _numWeights);
                    _allBiases.RemoveRange(idx * _numBiases, _numBiases);
                }

                bibiteData.obj.SetActive(false);
                _objectPool.Add(bibiteData);
                _bibiteDatas.Remove(bibiteData);
            }

            weightsBuffer.SetData(_allWeights);
            biasesBuffer.SetData(_allBiases);

            _neuralNetShader.SetBuffer(kernel, "weights", weightsBuffer);
            _neuralNetShader.SetBuffer(kernel, "biases", biasesBuffer);

            _toRemove.Clear();
        }

        if (_children.Count > 0)
        {
            _bibiteDatas.AddRange(_children);

            foreach (BibiteData child in _children)
            {
                _allWeights.AddRange(child.weights);
                _allBiases.AddRange(child.biases);
            }

            weightsBuffer.SetData(_allWeights);
            biasesBuffer.SetData(_allBiases);

            _neuralNetShader.SetBuffer(kernel, "weights", weightsBuffer);
            _neuralNetShader.SetBuffer(kernel, "biases", biasesBuffer);

            _children.Clear();
        }

        numBibites = _bibiteDatas.Count;
        objectPoolSize = _objectPool.Count;

        inputsBuffer.SetData(_allInputs);
        _neuralNetShader.SetBuffer(kernel, "inputs", inputsBuffer);

        _neuralNetShader.SetInt("numBibites", numBibites);

        int threadsPerGroup = 64;
        int batchSize = 2;
        int threadGroupsX = Mathf.CeilToInt((numBibites / (float)batchSize) / threadsPerGroup);
        _neuralNetShader.Dispatch(kernel, threadGroupsX, 1, 1);

        AsyncGPUReadback.Request(outputsBuffer, OnCompleteReadback);

        _allInputs.Clear();

        for (int i = 0; i < numBibites; i++)
        {
            BibiteData bibiteData = _bibiteDatas[i];
            Transform bibiteTransform = bibiteData.transform;
            Vector3 bibiteTransformRight = bibiteTransform.right;
            Bibite bibiteScript = bibiteData.script;
            Rigidbody2D bibiteRb = bibiteData.rb;
            float[] bibiteInputs = bibiteData.inputs;

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
                    bibiteInputs[inputIndex + 1] = hit.collider.CompareTag("Bibite") ? 1f : -1f;
                }
                else
                {
                    bibiteInputs[inputIndex] = -1f;
                    bibiteInputs[inputIndex + 1] = 0f;
                }
            }

            _allInputs.AddRange(bibiteInputs);

            int bestIdx = neuralNetOutputs[i];
            if (bestIdx == 0)
            {
                Vector3 direction = bibiteTransform.right * Mathf.Max(bibiteScript._energy / 75f, 0.1f);
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
        }
    }

    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.LogError("GPU readback error");
            return;
        }

        neuralNetOutputs = request.GetData<int>().ToArray();
    }

    private void Update()
    {
        float dt = Time.deltaTime;

        for (int i = 0; i < numBibites; i++)
        {
            BibiteData bibiteData = _bibiteDatas[i];
            Transform bibiteTransform = bibiteData.transform;
            Bibite bibiteScript = bibiteData.script;
            Rigidbody2D bibiteRb = bibiteData.rb;

            if ((bibiteScript._health <= 0 || bibiteScript._age >= bibiteScript._maxAge) && !_toRemove.Contains(bibiteData))
            {
                _toRemove.Add(bibiteData);
                continue;
            }

            bibiteScript._age += dt / 20f;
            if (!bibiteScript._matured && bibiteScript._age >= bibiteScript._maturityAge)
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
            if (bibiteTransform.position.y < divider.transform.position.y)
                bibiteScript._health -= dt * 4f;
        }
    }

    private void OnValidate()
    {
        if (Application.isPlaying)
        {
            if (kill5PercentOfBibites && numBibites >= 20)
            {
                int numToKill = Mathf.CeilToInt(numBibites / 20f);
                for (int i = 0; i < numToKill; i++)
                {
                    BibiteData bibiteData = _bibiteDatas[0];
                    bibiteData.obj.SetActive(false);
                    _objectPool.Add(bibiteData);
                    _bibiteDatas.RemoveAt(0);

                    _allWeights.RemoveRange(0, _numWeights);
                    _allBiases.RemoveRange(0, _numBiases);

                    numBibites--;
                }
                kill5PercentOfBibites = false;
            }

            else if (add100PooledBibites)
            {
                for (int i = 0; i < 100; i++)
                {
                    GameObject bibiteObj = Instantiate(_bibitePrefab, Vector3.zero, Quaternion.identity);
                    bibiteObj.SetActive(false);

                    _objectPool.Add(new BibiteData
                    {
                        obj = bibiteObj,
                        transform = bibiteObj.transform,
                        script = bibiteObj.GetComponent<Bibite>(),
                        rb = bibiteObj.GetComponent<Rigidbody2D>(),
                        inputs = new float[_numInputs]
                    });
                }
                add100PooledBibites = false;
            }

            else if (Time.time > 0.01f)
            {
                inputsBuffer = new(_numInputs * maxBibites, sizeof(float));
                weightsBuffer = new(_numWeights * maxBibites, sizeof(float));
                biasesBuffer = new(_numBiases * maxBibites, sizeof(float));
                outputsBuffer = new(maxBibites, sizeof(int));

                weightsBuffer.SetData(_allWeights.ToArray());
                biasesBuffer.SetData(_allBiases.ToArray());

                _neuralNetShader.SetBuffer(kernel, "inputs", inputsBuffer);
                _neuralNetShader.SetBuffer(kernel, "outputs", outputsBuffer);
                _neuralNetShader.SetBuffer(kernel, "weights", weightsBuffer);
                _neuralNetShader.SetBuffer(kernel, "biases", biasesBuffer);
            }
        }
    }

    private void AddBibites(BibiteData parentBibiteData, int numChildren)
    {
        Bibite parentScript = parentBibiteData.script;
        for (int i = 0; i < numChildren; i++)
        {
            if (_objectPool.Count > 0)
            {
                BibiteData poolBibiteData = _objectPool[0];
                GameObject poolBibiteObj = poolBibiteData.obj;
                Bibite poolBibiteScript = poolBibiteObj.GetComponent<Bibite>();
                Rigidbody2D poolBibiteRb = poolBibiteObj.GetComponent<Rigidbody2D>();

                poolBibiteScript._health = 100;
                poolBibiteScript._energy = 100;
                poolBibiteScript._age = 0;
                poolBibiteScript._matured = false;

                poolBibiteObj.transform.SetPositionAndRotation(
                    new(Random.Range(walls[3].transform.position.x * 1.05f, walls[2].transform.position.x * 1.05f), Random.Range(walls[1].transform.position.y * 1.05f, walls[0].transform.position.y * 1.05f), 0),
                    Quaternion.Euler(0, 0, Random.Range(0f, 360f)));

                MutateValues(parentScript, poolBibiteScript);
                poolBibiteObj.transform.localScale = Vector3.one * poolBibiteScript._size;
                poolBibiteRb.mass = poolBibiteScript._size;

                _neuralNet.ModifyParameters(parentBibiteData.weights, parentBibiteData.biases, out float[] newPoolWeights, out float[] newPoolBiases);
                poolBibiteData.weights = newPoolWeights;
                poolBibiteData.biases = newPoolBiases;

                poolBibiteObj.SetActive(true);
                _children.Add(poolBibiteData);
                _objectPool.RemoveAt(0);
            }

            else
            {
                Vector3 spawnPosition = new(Random.Range(walls[3].transform.position.x * 1.05f, walls[2].transform.position.x * 1.05f), Random.Range(walls[1].transform.position.y * 1.05f, walls[0].transform.position.y * 1.05f), 0);
                GameObject bibiteObj = Instantiate(_bibitePrefab, spawnPosition, Quaternion.Euler(0, 0, Random.Range(0f, 360f)));
                Bibite bibiteScript = bibiteObj.GetComponent<Bibite>();
                Rigidbody2D bibiteRb = bibiteObj.GetComponent<Rigidbody2D>();

                MutateValues(parentScript, bibiteScript);
                bibiteObj.transform.localScale = Vector3.one * bibiteScript._size;
                bibiteRb.mass = bibiteScript._size;

                _neuralNet.ModifyParameters(parentBibiteData.weights, parentBibiteData.biases, out float[] newWeights, out float[] newBiases);

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

                Debug.Log("Pool empty");
            }
        }

        parentScript._energy -= numChildren * 20;
    }

    private void MutateValues(Bibite parentScript, Bibite bibiteScript)
    {
        bibiteScript._generation = parentScript._generation + 1;
        bibiteScript._size = Random.Range(0, 2) == 1 ? Mathf.Max(parentScript._size - Random.Range(0.1f, 0.35f) + Random.Range(-0.05f, 0.05f), 0.05f) : parentScript._size - Random.Range(0.1f, 0.35f);
        bibiteScript._speed = Random.Range(0, 2) == 1 ? Mathf.Max(parentScript._speed + Random.Range(0.3f, -0.3f), 0.1f) : parentScript._speed;
        bibiteScript._rayLength = Random.Range(0, 2) == 1 ? Mathf.Max(parentScript._rayLength + Random.Range(-0.2f, 0.2f), 0f) : parentScript._rayLength;
        bibiteScript._maturityMinAge = Random.Range(0, 2) == 1 ? Mathf.Max(parentScript._maturityMinAge + Random.Range(-0.2f, 0.2f), 1f) : parentScript._maturityMinAge;
        bibiteScript._maturityMaxAge = Random.Range(0, 2) == 1 ? Mathf.Max(parentScript._maturityMaxAge + Random.Range(-0.2f, 0.2f), bibiteScript._maturityMinAge + 0.1f) : parentScript._maturityMaxAge;
        bibiteScript._maturityAge = Random.Range(bibiteScript._maturityMinAge, bibiteScript._maturityMaxAge);
    }
}