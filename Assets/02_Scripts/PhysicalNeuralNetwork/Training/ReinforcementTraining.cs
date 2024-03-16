using PhysicalNeuralNetwork.Character;
using TMPro;
using UnityEngine;

namespace PhysicalNeuralNetwork.Training
{
    public class ReinforcementTraining : MonoBehaviour
    {
        [SerializeField] private float learningRate = 0.001f;
        [SerializeField] private TMP_Text meanRewardText;
        [SerializeField] private TMP_Text currentRewardText;

        public uint MaxStep;
        private Axon[] _axons;
        private AgentController _agentController;

        private float _currentReward;
        private int _currentStep;
        private float _meanReward;
        private float _rewardChange;
        private void Start()
        {
            _rewardChange = 1f;
            _axons = FindObjectsOfType<Axon>();
            _agentController = FindObjectOfType<AgentController>();
            StartEpisode();
        }

        private void FixedUpdate()
        {
            _currentStep++;
            if(MaxStep > 0 &&  _currentStep > MaxStep)
                EndEpisode();
            
            meanRewardText.text = "Mean Reward:" + _meanReward.ToString("0.00");
            currentRewardText.text = "Reward:" + _currentReward.ToString("0.00");
        }

        private void StartEpisode()
        {
            _agentController.StartEpisode();
            _currentReward = 0f;
            _currentStep = 0;
        }
        public void SetReward(float reward)
        {
            _currentReward = reward;
        }

        public void AddReward(float reward)
        {
            _currentReward += reward;
        }

        public void EndEpisode()
        {
            var rewardDifference = _currentReward - _meanReward;
            var absRewardDifference = Mathf.Abs(rewardDifference);
            var absRewardChange = Mathf.Abs(_rewardChange);
            var rawAccuracyRate = absRewardDifference > absRewardChange ? 1f : absRewardDifference / absRewardChange == 0 ? 1 : absRewardChange;
            var accuracyRate = rewardDifference > 0 ? rawAccuracyRate : -rawAccuracyRate;
            foreach (var axon in _axons)
            {
                axon.RefreshWeightAndBiases(accuracyRate, learningRate);
            }

            _rewardChange = rewardDifference;
            _meanReward = _currentReward;
            StartEpisode();
        }
    }
}