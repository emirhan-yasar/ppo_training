using TMPro;
using UnityEngine;

namespace NeuronBasedNuralNetwork.Character
{
    public class AgentController : MonoBehaviour
    {
        [SerializeField] private float moveSpeed;
        [SerializeField] private Transform targetTransform;
        [SerializeField] private Brain brain;
        [SerializeField] private TMP_Text playerInfoText;
        
        private Vector3 _startPosition;
        private void Awake()
        {
            _startPosition = transform.position;
        }
        
        private void FixedUpdate()
        {
            var position = transform.position;
            var targetPosition = targetTransform.position;
            brain.SetInputNeurons(position.x, position.y, targetPosition.x, targetPosition.y);

            var outputs = brain.GetOutputValues();
            var moveVector = new Vector3(outputs[0], outputs[1], 0);
            
            position += Time.fixedDeltaTime * moveSpeed * moveVector.normalized;
            transform.position = position;
            
            playerInfoText.text = outputs[0].ToString("0.00") + " // " + outputs[1].ToString("0.00");
            Debug.DrawRay(position, moveVector.normalized, Color.green);
            
            brain.Train();
            if ((targetTransform.position - position).sqrMagnitude < 1)
            {
                brain.AddReward(1f);
                brain.EndEpisode();
            }
            
            var deltaPosition = (targetTransform.position - position).magnitude;
            if (Mathf.Abs(position.x) > 9 || Mathf.Abs(position.y) > 5)
            {
                brain.AddReward(-1f - deltaPosition / 9f);
                brain.EndEpisode();
            }
            //brain.AddReward(-1f / brain.MaxStep);
        }

        public void StartEpisode()
        {
            transform.position = _startPosition;
        }
    }
}