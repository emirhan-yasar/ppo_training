using TMPro;
using UnityEngine;

namespace PhysicalNeuralNetwork
{
    public class Receiver : MonoBehaviour
    {
        [SerializeField] private Neuron[] outputNeurons;
        [SerializeField] private TMP_Text text;
        public float Value => _value;
        private float _value;
        private void FixedUpdate()
        {
            if (outputNeurons.Length == 0) return;
            _value = 0;
            foreach (var outputNeuron in outputNeurons)
            {
                _value += outputNeuron.Value;
            }

            text.text = _value.ToString("0.000");
        }
    }
}