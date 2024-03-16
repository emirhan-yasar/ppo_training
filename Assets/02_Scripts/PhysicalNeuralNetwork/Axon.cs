using PhysicalNeuralNetwork;
using UnityEngine;

namespace PhysicalNeuralNetwork
{
    public class Axon : MonoBehaviour
    {
        private float _weight;
        private float _bias;

        private float _baseWeight;
        private float _baseBias;
        
        private LineRenderer _lineRenderer;
        private Neuron _inputNeuron;
        private Neuron _outputNeuron;
        
        private Transform _inputTransform;
        private Transform _outputTransform;
        
        
        private void Awake()
        {
            _lineRenderer = GetComponent<LineRenderer>();
            _lineRenderer.positionCount = 2;
            _baseWeight = Random.Range(-1f, 1f);
            _baseBias = Random.Range(-1f, 1f);

            _weight = _baseWeight + 0.001f;
            _bias = _baseBias + 0.001f;
            
            _lineRenderer.widthMultiplier = 0.025f * (Mathf.Abs(_weight));
            if (_lineRenderer.widthMultiplier < 0.0025f) _lineRenderer.widthMultiplier = 0f;
            
            _lineRenderer.startColor = _weight <= 0 ? Color.red : Color.cyan;
        }

        private void FixedUpdate()
        {
            _lineRenderer.SetPosition(0, _inputTransform.position);
            _lineRenderer.SetPosition(1, _outputTransform.position);
        }

        public void ConnectAxon(Neuron inputNeuron, Neuron outputNeuron)
        {
            _inputNeuron = inputNeuron;
            _outputNeuron = outputNeuron;

            _inputTransform = _inputNeuron.transform;
            _outputTransform = _outputNeuron.transform;
        }

        public float GetValue()
        {
            return _inputNeuron.ActivatedValue * _weight + _bias;
        }

        /// <summary>
        /// Accuracy Rate Must Be Between -1 and 1
        /// </summary>
        /// <param name="accuracyRate"></param>
        /// <param name="learningRate"></param>
        public void RefreshWeightAndBiases(float accuracyRate, float learningRate)
        {
            _baseWeight = Mathf.Lerp(_baseWeight, _weight, accuracyRate);
            _weight = _baseWeight + learningRate * Random.Range(-1f,1f);
            _weight = Mathf.Clamp(_weight, -1f, 1f);

            _baseBias = Mathf.Lerp(_baseBias, _bias, accuracyRate);
            _bias = _baseBias + learningRate * Random.Range(-1f,1f);
            _bias = Mathf.Clamp(_bias, -1f, 1f);
            
            if(Mathf.Abs(accuracyRate) > 0.0001f) return;
            _baseWeight = _weight;
            _baseBias = _bias;
            
        }
    }
}