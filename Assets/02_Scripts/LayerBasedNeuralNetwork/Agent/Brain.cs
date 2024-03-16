using System;
using System.Collections.Generic;
using System.Linq;
using LayerBasedNeuralNetwork.Network;
using UnityEngine;

namespace LayerBasedNeuralNetwork.Agent
{
    public class Brain : MonoBehaviour
    {
        [Header("Neural Network Parameters")]
        [SerializeField] private uint inputLayerSize;
        [SerializeField] private uint outputLayerSize;
        [SerializeField] private uint[] hiddenLayerSizes;
        [SerializeField] private ComputeShader layerComputeShader;
        [SerializeField] private ComputeDevice computeDevice;
        
        [Header("Agent Parameters")] 
        [SerializeField] private AgentController agentController;
        [SerializeField] private BehaviorType behaviorType;

        public float[] Actions => _actions;

        private NeuralNetwork _neuralNetwork;
        private float[] _observations;
        private float[] _actions;
        private void Awake()
        {
            var sizes = hiddenLayerSizes.ToList();
            sizes.Insert(0,inputLayerSize);
            sizes.Add(outputLayerSize);
            _neuralNetwork = new NeuralNetwork(layerComputeShader, computeDevice, sizes.ToArray());

            _observations = new float[inputLayerSize];
            _actions = new float[outputLayerSize];
        }

        public void Run()
        {
            switch(behaviorType)
            {
                case BehaviorType.Default:
                    break;
                case BehaviorType.Heuristic:
                    agentController.Heuristic();
                    break;
                case BehaviorType.Interface:
                    agentController.AddObservations();
                    ReceiveAction();
                    break;
                case BehaviorType.Learning:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
        private void ReceiveAction()
        {
            var output = _neuralNetwork.Evaluate(_observations);
            if (output.Length != _actions.Length)
            {
                print("Number of actions is not equal to the outputLayerSize");
                return;
            }
            for (var i = 0; i < output.Length; i++)
                _actions[i] = output[i];
        }
        public float[] Evaluate(params float[] inputs)
        {
            return _neuralNetwork.Evaluate(inputs);
        }
        public void SetObservations(params float[] observations)
        {
            if (observations.Length != _observations.Length)
            {
                print("Number of observations is not equal to the inputLayerSize");
                return;
            }
            for (var i = 0; i < observations.Length; i++)
                _observations[i] = observations[i];
        }

        private void OnDisable()
        {
            _neuralNetwork.OnDisable();
        }
    }
}