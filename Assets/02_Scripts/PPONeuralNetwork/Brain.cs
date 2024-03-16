using System.Linq;
using TMPro;
using UnityEngine;

namespace PPONeuralNetwork
{
    public class Brain : MonoBehaviour
    {
        [SerializeField] private int[] layerSizes;

        private Neuron[] _allNeurons;
        private Neuron[] _inputNeurons;
        private Neuron[] _outputNeurons;

        private void Awake()
        {
            
            if (layerSizes.Length < 2 || layerSizes[0] < 1 || layerSizes[^1] < 1)
            {
                print("Not Proper Layers");
                return;
            }
            
            // Create Neurons
            var allNeuronCount = layerSizes.Sum();
            _allNeurons = new Neuron[allNeuronCount];
            
            for (int i = 0; i < layerSizes.Length; i++)
            {
                var neuronCount = layerSizes[i];
                for (int j = 0; j < neuronCount; j++)
                {
                    var neuronIndex = i == 0 ? j : j + layerSizes.Take(i).Sum();
                    
                    //Create a new neuron and assign it its input neurons
                    _allNeurons[neuronIndex] = new Neuron(i == 0 ? null : NeuronsAtLayer(i - 1),neuronIndex >= allNeuronCount - layerSizes[^1]);
                }
            }

            _inputNeurons = new Neuron[layerSizes[0]];
            for (int i = 0; i < _inputNeurons.Length; i++)
                _inputNeurons[i] = _allNeurons[i];

            _outputNeurons = new Neuron[layerSizes[^1]];
            for (int i = 0; i < _outputNeurons.Length; i++)
                _outputNeurons[i] = _allNeurons[_allNeurons.Length - _outputNeurons.Length + i];

        }
        

        private Neuron[] NeuronsAtLayer(int layerIndex)
        {
            var neuronCount = layerSizes[layerIndex];
            var neuronStartIndex = 0;
            for (int i = 0; i < layerIndex; i++)
                neuronStartIndex += layerSizes[i];

            var neuronArray = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
                neuronArray[i] = _allNeurons[neuronStartIndex + i];

            return neuronArray;
        }

        public float[] Evaluate(params float[] inputValues)
        {
            SetInputNeurons(inputValues);
            foreach (var neuron in _allNeurons)
            {
                neuron.Activate();
            }
            return GetOutputValues();
        }
        private void SetInputNeurons(params float[] values)
        {
            if (values.Length != _inputNeurons.Length)
            {
                print("Input Neuron Length is not equal to Input");
                return;
            }

            for (int i = 0; i < values.Length; i++)
            {
                _inputNeurons[i].Value = values[i];
            }
        }

        private float[] GetOutputValues()
        {
            var values = new float[_outputNeurons.Length];
            for (int i = 0; i < _outputNeurons.Length; i++)
                values[i] = _outputNeurons[i].Value;

            return values;
        }

        public void Train()
        {
        }

        public void AddReward(float value)
        {
        }

        public void SetReward(float value)
        {
        }

        public void EndEpisode()
        {
        }
    }
}