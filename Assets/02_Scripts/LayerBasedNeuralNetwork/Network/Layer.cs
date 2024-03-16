using UnityEngine;

namespace LayerBasedNeuralNetwork.Network
{
    public class Layer
    {
        private readonly uint _numNeuronsIn;
        private readonly uint _numNeuronsOut;
        //private readonly float[,] _weights;
        private readonly float[] _weights;
        private readonly float[] _biases;
        
        private ComputeDevice _computeDevice;
        private bool _isOutputLayer;
        
        private readonly ComputeShader _layerComputeShader;
        private readonly ComputeBuffer _weightBuffer;
        private readonly ComputeBuffer _biasesBuffer;
        private readonly ComputeBuffer _inputBuffer;
        private readonly ComputeBuffer _outputBuffer;
        private readonly int _csHiddenKernelID;
        private readonly int _csOutputKernelID;

        public Layer(uint numNeuronsIn, uint numNeuronsOut, ComputeShader layerComputeShader, ComputeDevice computeDevice, bool isOutputLayer)
        {
            _numNeuronsIn = numNeuronsIn;
            _numNeuronsOut = numNeuronsOut;

            //_weights = new float[numNeuronsOut,numNeuronsIn]; // numOfRows = numNeuronsOut, numOfColumns = numNeuronsIn
            _weights = new float[numNeuronsOut * numNeuronsIn];
            _biases = new float[numNeuronsOut];
            RandomlyInitializeLayer();

            _isOutputLayer = isOutputLayer;
            
            _computeDevice = computeDevice;
            if(_computeDevice == ComputeDevice.CPU) return;
            
            _layerComputeShader = layerComputeShader;
            _layerComputeShader.SetInt("num_neurons_in",(int)_numNeuronsIn);
            _layerComputeShader.SetInt("num_neurons_out", (int)_numNeuronsOut);
            _csHiddenKernelID = _layerComputeShader.FindKernel("CSHiddenLayer");
            _csOutputKernelID = _layerComputeShader.FindKernel("CSOutputLayer");

            _weightBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _inputBuffer = new ComputeBuffer((int)_numNeuronsIn, sizeof(float));
            _outputBuffer = new ComputeBuffer((int)_numNeuronsOut, sizeof(float));
            
            _weightBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);

            var kernelID = !_isOutputLayer ? _csHiddenKernelID : _csOutputKernelID;
            _layerComputeShader.SetBuffer(kernelID, "inputs", _inputBuffer);
            _layerComputeShader.SetBuffer(kernelID, "weights", _weightBuffer);
            _layerComputeShader.SetBuffer(kernelID, "biases", _biasesBuffer);
            _layerComputeShader.SetBuffer(kernelID, "values", _outputBuffer);
        }

        public float[] CalculateOutputs(float[] inputs)
        {
            var layerOutputs = new float[_numNeuronsOut];
            /*for (var numOut = 0; numOut < _numNeuronsOut; numOut++)
            {
                var output = _biases[numOut];
                for (var numIn = 0; numIn < _numNeuronsIn; numIn++)
                {
                    output += _weights[numOut, numIn] * inputs[numIn];
                }
                output = ActivationFunctions.Sigmoid(output);
                layerOutputs[numOut] = output;
            }*/
            if (_computeDevice == ComputeDevice.CPU)
            {
                for (var numOut = 0; numOut < _numNeuronsOut; numOut++)
                {
                    var output = _biases[numOut];
                    for (var numIn = 0; numIn < _numNeuronsIn; numIn++)
                    {
                        output += _weights[numOut * _numNeuronsIn + numIn] * inputs[numIn];
                    }
                    output = !_isOutputLayer ? ActivationFunctions.Swish(output) : ActivationFunctions.LinearClip(output);
                    layerOutputs[numOut] = output;
                }

                return layerOutputs;
            }
            _inputBuffer.SetData(inputs);
            _weightBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);
            
            if (!_isOutputLayer)
            {
                _layerComputeShader.SetBuffer(_csHiddenKernelID, "inputs", _inputBuffer);
                _layerComputeShader.SetBuffer(_csHiddenKernelID, "weights", _weightBuffer);
                _layerComputeShader.SetBuffer(_csHiddenKernelID, "biases", _biasesBuffer);
                _layerComputeShader.SetBuffer(_csHiddenKernelID, "values", _outputBuffer);
                _layerComputeShader.Dispatch(_csHiddenKernelID, (int)_numNeuronsOut / 8, 1, 1);
            }
            else
            {
                _layerComputeShader.SetBuffer(_csOutputKernelID, "inputs", _inputBuffer);
                _layerComputeShader.SetBuffer(_csOutputKernelID, "weights", _weightBuffer);
                _layerComputeShader.SetBuffer(_csOutputKernelID, "biases", _biasesBuffer);
                _layerComputeShader.SetBuffer(_csOutputKernelID, "values", _outputBuffer);
                _layerComputeShader.Dispatch(_csOutputKernelID, (int)_numNeuronsOut / 2, 1, 1);
            }

            _outputBuffer.GetData(layerOutputs);
            return layerOutputs;
        }

        private void RandomlyInitializeLayer()
        {
            /*for (var row = 0; row < _weights.GetLength(0); row++)
            for (var column = 0; column < _weights.GetLength(1); column++)
            {
                _weights[row, column] = Random.Range(-2f, 2f);
            }*/
            for (var i = 0; i < _weights.Length; i++)
                _weights[i] = Random.Range(-2f, 2f);
                //_weights[i] = (i % 10 - 3.5f) / 5f;

            for (var i = 0; i < _biases.Length; i++)
                _biases[i] = Random.Range(-2f, 2f);
                //_biases[i] = (i % 10 - 2.5f) / 5f;
        }

        public void OnDisable()
        {
            if(_computeDevice == ComputeDevice.CPU) return;
            _biasesBuffer.Release();
            _weightBuffer.Release();
            _inputBuffer.Release();
            _outputBuffer.Release();
        }

    }
}