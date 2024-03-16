using PPONeuralNetwork.PPO;
using TMPro;
using UnityEngine;

namespace PPONeuralNetwork.Character
{
    public class AgentController : MonoBehaviour
    {
        [SerializeField] private float moveSpeed;
        [SerializeField] private Transform targetTransform;
        [SerializeField] private Brain brain;
        [SerializeField] private TMP_Text playerInfoText;

        [Header("Training")] 
        [SerializeField] private int maxStep;
        [SerializeField] private float learningRate = 0.0003f;
        [SerializeField] private float gamma = 0.98f;
        [SerializeField] private int numEpochs = 3;
        [SerializeField] private int numBatches = 10;
        [SerializeField] private float clipEpsilon = 0.2f;
        
        
        private Vector3 _startPosition;
        private PPOTrainer _ppoTrainer;
        
        
        private void Awake()
        {
            _startPosition = transform.position;
            _ppoTrainer = new PPOTrainer(this, brain, gamma, numEpochs, numBatches, clipEpsilon);
        }
        
        private void FixedUpdate()
        {
            var position = transform.position;
            var targetPosition = targetTransform.position;

            var state = new [] { position.x, position.y, targetPosition.x, targetPosition.y };
            
            
            var actions = brain.Evaluate(state);

            var nextPosition = position + new Vector3(actions[0], actions[1], 0);
            
            transform.position = nextPosition;
            
            var (done, reward) = EvaluateDeltaReward(nextPosition, targetPosition);
            _ppoTrainer.AddReward(reward);
            
            var nextState = new [] { nextPosition.x, nextPosition.y, targetPosition.x, targetPosition.y };


            var experience = new Experience(state, actions, reward, nextState, done);
            _ppoTrainer.AddExperience(experience);
            
            if (done)
            {
                _ppoTrainer.EndEpisode();
                return;
            }
            
            /*_ppoTrainer.AddReward(-1f/maxStep);
            if (Mathf.Abs(position.x) > 9f || Mathf.Abs(position.y) > 5f)
            {
                _ppoTrainer.SetReward(-1f);
                _ppoTrainer.EndEpisode();
                return;
            }

            if ((position - targetPosition).sqrMagnitude < 1f)
            {
                _ppoTrainer.SetReward(2f);
                _ppoTrainer.EndEpisode();
                return;
            }*/



        }

        private (bool endEpisode, float deltaReward) EvaluateDeltaReward(Vector3 position, Vector3 targetPosition)
        {
            var currentReward = _ppoTrainer.CumReward;
            var newReward = currentReward;
            newReward += -1f / maxStep;

            if (Mathf.Abs(position.x) > 9f || Mathf.Abs(position.y) > 5f)
            {
                newReward = -1f;
                return (true, newReward - currentReward);
            }

            if ((position - targetPosition).sqrMagnitude < 1f)
            {
                newReward = 2f;
                return (true, newReward - currentReward);
            }

            return (false, newReward - currentReward);
        }

        public void StartEpisode()
        {
            transform.position = _startPosition;
        }
    }
}