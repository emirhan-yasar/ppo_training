using PhysicalNeuralNetwork.Training;
using UnityEngine;

namespace PhysicalNeuralNetwork.Character
{
    public class AgentController : MonoBehaviour
    {
        [SerializeField] private Receiver receiverX;
        [SerializeField] private Receiver receiverY;
        [SerializeField] private float moveSpeed;
        [SerializeField] private Transform targetTransform;

        private ReinforcementTraining _reinforcementTraining;
        private Vector3 _startPosition;
        private void Awake()
        {
            _reinforcementTraining = FindObjectOfType<ReinforcementTraining>();
            _startPosition = transform.position;
        }

        private void FixedUpdate()
        {
            var moveDirection = new Vector3(receiverX.Value, receiverY.Value, 0);
            transform.position += Time.fixedDeltaTime * moveSpeed * moveDirection;
            var position = transform.position;
            Debug.DrawRay(position, moveDirection.normalized, Color.green);
            if ((targetTransform.position - position).sqrMagnitude < 1)
            {
                _reinforcementTraining.AddReward(1f);
                _reinforcementTraining.EndEpisode();
            }

            if (Mathf.Abs(position.x) > 9 || Mathf.Abs(position.y) > 5)
            {
                _reinforcementTraining.AddReward(-1f);
                _reinforcementTraining.EndEpisode();
            }
            _reinforcementTraining.AddReward(1f / _reinforcementTraining.MaxStep);
            //_reinforcementTraining.AddReward(-1f/_reinforcementTraining.MaxStep);
        }

        public void StartEpisode()
        {
            transform.position = _startPosition;
        }
        
    }
}