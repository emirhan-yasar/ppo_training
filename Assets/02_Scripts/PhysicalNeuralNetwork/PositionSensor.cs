using UnityEngine;

namespace PhysicalNeuralNetwork
{
    public class PositionSensor : MonoBehaviour
    {
        [SerializeField] private Neuron[] connectedNeurons;
        
        [SerializeField] private Transform character;
        private void FixedUpdate()
        {
            if(connectedNeurons.Length == 0) return;
            var position = character.position;

            connectedNeurons[0].SetValue(position.x);
            connectedNeurons[1].SetValue(position.y);
        }
    }
}