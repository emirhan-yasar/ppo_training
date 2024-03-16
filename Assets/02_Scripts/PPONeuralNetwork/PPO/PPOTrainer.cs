

using System.Collections.Generic;
using System.Linq;
using PPONeuralNetwork.Character;
using UnityEngine;

namespace PPONeuralNetwork.PPO
{
    public class PPOTrainer
    {
        public float CumReward => _cumReward;

        private float _gamma;
        private int _numEpochs;
        private int _numBatches;
        private float _clipEpsilon;
        
        private float _oldLogProbability = 0f; // You need to maintain the log probability of the previous action

        private float _cumReward;
        private AgentController _agentController;
        private Brain _brain;
        
        private List<Experience> _experienceList;

        public PPOTrainer(AgentController agentController, Brain brain, float gamma, int numEpochs, int numBatches, float clipEpsilon)
        {
            _agentController = agentController;
            _gamma = gamma;
            _numEpochs = numEpochs;
            _numBatches = numBatches;
            _clipEpsilon = clipEpsilon;
        }

        public void StartEpisode()
        {
            _experienceList.Clear();
            _agentController.StartEpisode();
        }
        
        public void AddReward(float reward)
        {
            _cumReward += reward;
        }

        public void SetReward(float reward)
        {
            _cumReward = reward;
        }

        public void EndEpisode()
        {
            //Update Network
            TrainPPO();
            StartEpisode();
        }

        public void AddExperience(Experience experience)
        {
            _experienceList.Add(experience);
        }

        private void TrainPPO()
        {
            var states = _experienceList.Select(exp => exp.State).ToArray();
            var actions = _experienceList.Select(exp => exp.Actions).ToArray();
            var rewards = _experienceList.Select(exp => exp.Reward).ToArray();
            var nextStates = _experienceList.Select(exp => exp.NextState).ToArray();
            var dones = _experienceList.Select(exp => exp.Done).ToArray();
            
            // Compute advantages and returns
            float[] advantages = ComputeAdvantages(states, rewards, dones);
            float[] returns = ComputeReturns(rewards, dones);
            
            // PPO optimization
            for (int iteration = 0; iteration < _numEpochs; iteration++)
            {
                // Shuffle the experiences
                ShuffleExperiences(states, actions, advantages, returns, nextStates, dones);

                // Mini-batch optimization
                var miniBatchSize = 1;
                for (int batchStart = 0; batchStart < _numBatches; batchStart += miniBatchSize)
                {
                    int batchEnd = Mathf.Min(batchStart + miniBatchSize, _experienceList.Count);

                    // Select mini-batch
                    var miniBatchStates = states.Skip(batchStart).Take(batchEnd - batchStart).ToArray();
                    var miniBatchActions = actions.Skip(batchStart).Take(batchEnd - batchStart).ToArray();
                    var miniBatchAdvantages = advantages.Skip(batchStart).Take(batchEnd - batchStart).ToArray();
                    var miniBatchReturns = returns.Skip(batchStart).Take(batchEnd - batchStart).ToArray();
                    var miniBatchNextStates = nextStates.Skip(batchStart).Take(batchEnd - batchStart).ToArray();
                    var miniBatchDones = dones.Skip(batchStart).Take(batchEnd - batchStart).ToArray();

                    // Compute PPO loss
                    float loss = ComputePPOLoss(miniBatchStates, miniBatchActions, miniBatchAdvantages, miniBatchReturns, miniBatchNextStates, miniBatchDones);

                    // Perform optimization step (adjust this based on your neural network implementation)
                    //neuralNetwork.Optimize(loss);
                }
            }
            
        }

        private float[] ComputeAdvantages(float[][] states, float[] rewards, bool[] dones)
        {
            float[] advantages = new float[rewards.Length];
            float runningAdvantage = 0;

            for (int t = rewards.Length - 1; t >= 0; t--)
            {
                runningAdvantage = rewards[t] + _gamma * (1 - System.Convert.ToSingle(dones[t])) * runningAdvantage;
                advantages[t] = runningAdvantage - _brain.Evaluate(states[t])[0]; // I Am Not Sure
            }

            // Optionally, you might want to normalize advantages
            advantages = NormalizeAdvantages(advantages);

            return advantages;
        }

        private float[] ComputeReturns(float[] rewards, bool[] dones)
        {
            float[] returns = new float[rewards.Length];
            float runningReturn = 0;

            for (int t = rewards.Length - 1; t >= 0; t--)
            {
                runningReturn = rewards[t] + _gamma * (1 - System.Convert.ToSingle(dones[t])) * runningReturn;
                returns[t] = runningReturn;
            }

            return returns;
        }

        private float[] NormalizeAdvantages(float[] advantages)
        {
            // Optionally, you might want to normalize advantages to have zero mean and unit variance
            float mean = advantages.Average();
            float stdDev = Mathf.Sqrt(advantages.Select(adv => (adv - mean) * (adv - mean)).Sum() / advantages.Length);

            return advantages.Select(adv => (adv - mean) / (stdDev + 1e-8f)).ToArray();
        }
        
        private float ComputePPOLoss(float[][] states, float[][] actions, float[] advantages, float[] returns, float[][] nextStates, bool[] dones)
        {
            float policyLoss = 0f;
            float valueLoss = 0f;

            for (int t = 0; t < states.Length; t++)
            {
                // Forward pass to get current policy probabilities and value estimate
                float[] currentActionProbs = _brain.Evaluate(states[t]); 
                float currentValue = currentActionProbs[0];

                // Calculate the log probability of the taken action
                float logProbability = Mathf.Log(currentActionProbs[0]); // Replace with the actual log probability calculation

                // Compute the advantage and value target
                float advantage = advantages[t];
                float valueTarget = returns[t];

                // PPO policy loss
                float ratio = Mathf.Exp(logProbability - _oldLogProbability);
                float clippedRatio = Mathf.Clamp(ratio, 1 - _clipEpsilon, 1 + _clipEpsilon);
                float policyAdvantage = advantage * ratio;
                float clippedPolicyAdvantage = advantage * clippedRatio;
                float policyLoss_t = -Mathf.Min(policyAdvantage, clippedPolicyAdvantage);

                // PPO value loss
                float valueLoss_t = 0.5f * Mathf.Pow(currentValue - valueTarget, 2);

                // Accumulate losses
                policyLoss += policyLoss_t;
                valueLoss += valueLoss_t;

                // Update old log probability for the next iteration
                _oldLogProbability = logProbability;
            }

            // Combine policy and value losses
            float ppoLoss = policyLoss + valueLoss;

            return ppoLoss;
        }
        
        private void ShuffleExperiences(float[][] states, float[][] actions, float[] advantages, float[] returns, float[][] nextStates, bool[] dones)
        {
            int numExperiences = states.Length;
            System.Random random = new System.Random();

            for (int i = numExperiences - 1; i > 0; i--)
            {
                // Generate a random index to swap with the current index
                int randomIndex = random.Next(0, i + 1);

                // Swap states
                Swap(ref states[i], ref states[randomIndex]);

                // Swap actions
                Swap(ref actions[i], ref actions[randomIndex]);

                // Swap advantages
                Swap(ref advantages[i], ref advantages[randomIndex]);

                // Swap returns
                Swap(ref returns[i], ref returns[randomIndex]);

                // Swap nextStates
                Swap(ref nextStates[i], ref nextStates[randomIndex]);

                // Swap dones
                Swap(ref dones[i], ref dones[randomIndex]);
            }
        }

        private static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }
        
        

    }
}