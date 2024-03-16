using JetBrains.Annotations;
using UnityEngine;

namespace PPONeuralNetwork
{
    public class Neuron
    {
        public float Value { 
            get => _value;
            set => _value = value;
        }
        
        private float _value;
        private Neuron[] _inputNeurons;
        private float[] _weights;
        private float[] _baseWeights;
        
        private float _bias;
        private float _baseBias;
        private bool _isOutputNeuron;

        public Neuron([CanBeNull] Neuron[] inputNeurons, bool isOutputNeuron)
        {
            _inputNeurons = inputNeurons;
            if(_inputNeurons == null) return;
            _baseWeights = new float[_inputNeurons.Length];
            _weights = new float[_inputNeurons.Length];
            
            // Initialize Randomly:
            for (int i = 0; i < _baseWeights.Length; i++)
            {
                var weightValue = Random.Range(-1f, 1f);
                _baseWeights[i] = weightValue;
                _weights[i] = weightValue;
                //_weights[i] = weightValue + Random.Range(-1, 2) * learningRate;
            }

            
            
            _baseBias = Random.Range(-1f, 1f);
            _bias = _baseBias;
            //_bias = _baseBias + Random.Range(-1, 2) * learningRate;
            _isOutputNeuron = isOutputNeuron;
        }

        public void Activate()
        {
            if(_inputNeurons == null) return;
            
            // Can be used LINQ expression (This is more efficient)
            float inputValue = 0;
            for (var i = 0; i < _inputNeurons.Length; i++)
            {
                inputValue += _inputNeurons[i].Value * _weights[i];
            }

            inputValue += _bias;

            if (!_isOutputNeuron)
            {
                // Swish function
                var sigmoid = Sigmoid(inputValue);
                _value = sigmoid * inputValue;
                //_value = sigmoid;
                return;
            }

            
            //_value = SigmoidExpended(inputValue);
            //Deterministic Continuous
            var val = Mathf.Clamp(inputValue, -3f, 3f);
            _value = val / 3f;
            
        }

        public void Train(float accuracyRate, float learningRate, float momentum)
        {
            if(_weights == null) return;
            //Update Weights

            var accuracyFactor = accuracyRate >= 0f ? 1f / (1f + accuracyRate * 9) : 1f - accuracyRate * 2;
            //var accuracyFactor = accuracyRate >= 0f ? 1f + accuracyRate : 1f - accuracyRate;

            for (int i = 0; i < _weights.Length; i++)
            {
                //var deltaWeight = _weights[i] - _baseWeights[i];
                //var weightMultiplier = deltaWeight >= 0 ? 1f : -1f;
                //_baseWeights[i] = accuracyRate >= 0f ? _weights[i] : _baseWeights[i];
                //_weights[i] = _baseWeights[i] + learningRate * weightMultiplier * accuracyRate + deltaWeight * 5f;
                //_weights[i] = Mathf.Clamp(_weights[i], -1f, 1f);//TODO: I am not sure that I should limit this
                
                _baseWeights[i] = Mathf.Lerp(_baseWeights[i], _weights[i], accuracyRate);
                _weights[i] = _baseWeights[i] + learningRate * Random.Range(-1f,1f) * accuracyFactor + momentum;
            }
            
            //var deltaBias = _bias - _baseBias;
            //var biasMultiplier = deltaBias >= 0 ? 1f : -1f;
            //_baseBias = accuracyRate >= 0f ? _bias : _baseBias;
            //_bias = _baseBias + learningRate * biasMultiplier * accuracyRate + deltaBias * 5f;
            //_bias = Mathf.Clamp(_bias, -1f, 1f); //TODO: I am not sure that I should limit this
            
            _baseBias = Mathf.Lerp(_baseBias, _bias, accuracyRate);
            _bias = _baseBias + learningRate * Random.Range(-1f,1f) * accuracyFactor * momentum;
            
            if(Mathf.Abs(accuracyRate) > 0.0001f) return;
            
            _baseWeights = (float[])_weights.Clone();
            _baseBias = _bias;
        }

        private static float Sigmoid(float value)
        {
            return 1f / (1f + Mathf.Exp(-value));
        }
        
        private float SigmoidExpended(float value)
        {
            return 2f / (1f + Mathf.Exp(-value)) - 1f;
        }

        private float TanH(float value)
        {
            var exp = Mathf.Exp(value);
            var mexp = Mathf.Exp(-value);
            return (exp + mexp) / (exp + mexp);
        }
    }
}