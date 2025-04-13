using UnityEngine;

public class Bibite : MonoBehaviour
{
    public int _generation = 0;

    [Header("Properties")]
    public float _size = 1;
    public float _speed = 10;

    [Header("Vitals")]
    public float _health = 100;
    public float _energy = 100;

    [Header("Age")]
    public float _age = 0;
    public float _maxAge = 5;
    public float _maturityAge;
    [HideInInspector] public bool _matured = false;
    public float _maturityMinAge = 3;
    public float _maturityMaxAge = 5;

    [Space]

    public float _rayLength = 10f;

    private void Start()
    {
        _maturityAge = Random.Range(_maturityMinAge, _maturityMaxAge);
        transform.localScale = Vector3.one * _size;
    }
}