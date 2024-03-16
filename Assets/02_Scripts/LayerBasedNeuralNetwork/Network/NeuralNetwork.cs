using UnityEngine;

namespace LayerBasedNeuralNetwork.Network
{
    public class NeuralNetwork
    {
        private readonly Layer[] _layers;

        public NeuralNetwork(ComputeShader layerComputeShader,ComputeDevice computeDevice, params uint[] layerSizes)
        {
            // Not include input layer
            _layers = new Layer[layerSizes.Length - 1];

            for (var i = 0; i < _layers.Length - 1; i++)
            {
                _layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], layerComputeShader, computeDevice, false);
            }
            _layers[^1] = new Layer(layerSizes[^2], layerSizes[^1], layerComputeShader, computeDevice, true);
        }

        public float[] Evaluate(float[] inputs)
        {
            foreach (var layer in _layers)
            {
                inputs = layer.CalculateOutputs(inputs);
            }

            return inputs;
        }

        public void OnDisable()
        {
            foreach (var layer in _layers)
            {
                layer.OnDisable();
            }
        }
    }
}