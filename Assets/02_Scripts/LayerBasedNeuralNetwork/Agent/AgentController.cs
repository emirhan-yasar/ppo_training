using UnityEngine;

namespace LayerBasedNeuralNetwork.Agent
{
    public class AgentController : MonoBehaviour
    {
        [SerializeField] private float moveSpeed;
        [SerializeField] private Brain brain;
        [SerializeField] private Transform targetTransform;
        
        private Transform _transform;
        private void Awake()
        {
            _transform = transform;
        }

        private void FixedUpdate()
        {
            brain.Run();
            var actions = brain.Actions;
            var moveVector = new Vector3(actions[0], actions[1], 0);
            transform.position += Time.fixedDeltaTime * moveSpeed * moveVector;
        }

        public void AddObservations()
        {
            var position = _transform.position;
            var targetPosition = targetTransform.position;

            brain.SetObservations(position.x, position.y, targetPosition.x, targetPosition.y);
        }
        public void Heuristic()
        {
            brain.Actions[0] = Input.GetAxis("Horizontal");
            brain.Actions[1] = Input.GetAxis("Vertical");
        }
    }
}