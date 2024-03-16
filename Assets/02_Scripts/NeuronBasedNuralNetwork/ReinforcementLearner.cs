using NeuronBasedNuralNetwork.Character;
using UnityEngine;

namespace NeuronBasedNuralNetwork
{
    public class ReinforcementLearner
    {
        public float TotalReward => _totalReward;
        public float CurrentReward => _currentReward;

        public int MaxStep
        {
            get => _maxStep;
            set => _maxStep = value;
        }

        private float _learningRate;
        private float _totalReward;
        private float _currentReward;
        private float _lastDeltaReward;
        private int _currentStep;
        private int _maxStep;

        private float _momentum;
        
        private AgentController _agentController;
        private Neuron[] _neurons;

        public ReinforcementLearner(AgentController agentController, Neuron[] neurons, int maxStep, float learningRate)
        {
            _agentController = agentController;
            _neurons = neurons;
            _maxStep = maxStep;
            _learningRate = learningRate;
            
            _totalReward = float.NegativeInfinity;
            _momentum = 1f;
        }

        public void Train()
        {
            _currentStep++;
            if(_currentStep >= _maxStep)
                EndEpisode();
        }

        private void StartEpisode()
        {
            _agentController.StartEpisode();
            _currentReward = 0;
            _currentStep = 0;
        }
        public void AddReward(float value)
        {
            _currentReward += value;
        }

        public void SetReward(float value)
        {
            _currentReward = value;
        }

        public void EndEpisode()
        {
            //TODO: THERE IS SOMETHING WRONG HERE
            /*if (float.IsNegativeInfinity(_totalReward))
            {
                foreach (var neuron in _neurons)
                {
                    neuron.Train(0f, _learningRate);
                }

                _totalReward = _currentReward;
                return;
            }*/
            
            var deltaReward = _currentReward - _totalReward;
            var absRewardDifference = Mathf.Abs(deltaReward);
            var absLastDeltaReward = Mathf.Abs(_lastDeltaReward);
            //var rawAccuracyRate = absRewardDifference >= absLastDeltaReward ? 1f : absRewardDifference / absLastDeltaReward;
            var rawAccuracyRate = absRewardDifference > absLastDeltaReward ? 1f : absRewardDifference / absLastDeltaReward == 0 ? 1 : absLastDeltaReward;

            var accuracyRate = deltaReward > 0 ? rawAccuracyRate : -rawAccuracyRate;

            if (accuracyRate < 0)
            {
                _momentum += _learningRate;
            }
            else
            {
                _momentum = 1f;
            }
            
            foreach (var neuron in _neurons)
            {
                neuron.Train(accuracyRate, _learningRate, _momentum);
            }

            // TODO ???????
            if (accuracyRate >= -0.0001f)
            {
                _lastDeltaReward = deltaReward;
                _totalReward = _currentReward;
            }
            
            StartEpisode();

            /*
            var deltaReward = _currentReward - _totalReward;
            var absRewardDifference = Mathf.Abs(deltaReward);
            var absLastDeltaReward = Mathf.Abs(_lastDeltaReward);
            var rawAccuracyRate = absRewardDifference > absLastDeltaReward ? 1f : absRewardDifference / absLastDeltaReward == 0 ? 1 : absLastDeltaReward;
            var accuracyRate = deltaReward > 0 ? rawAccuracyRate : -rawAccuracyRate;

            foreach (var neuron in _neurons)
            {
                neuron.Train(accuracyRate, _learningRate);
            }
            _lastDeltaReward = deltaReward;
            _totalReward = _currentReward;
            StartEpisode();*/
            
        }
    }
}