using TMPro;
using UnityEngine;

namespace PhysicalNeuralNetwork
{
    public class Neuron : MonoBehaviour
    {
        [SerializeField] private Neuron[] inputNeurons;
        [SerializeField] private Axon axonPrefab;
        [SerializeField] private TMP_Text text;
        public float Value => _value;
        public float ActivatedValue => ActivationFunction(_value);
        
        private float _value;
        private Transform _transform;
        private Axon[] _axons;
        private void Awake()
        {
            _transform = transform;
            
            if(inputNeurons.Length == 0) return;
            _axons = new Axon[inputNeurons.Length];
            for (var i = 0; i < inputNeurons.Length; i++)
            {
                var inputNeuron = inputNeurons[i];
                var axon = Instantiate(axonPrefab, _transform);
                axon.transform.localPosition = Vector3.zero;
                axon.ConnectAxon(inputNeuron, this);
                _axons[i] = axon;
            }
        }

        private void FixedUpdate()
        {
            text.text = _value.ToString("0.00");
            //if(Mathf.Abs(_value) < 0.0001f) return;
            //_value = DecayFunction(_value);
            if(_axons == null || _axons.Length == 0) return;
            _value = 0;
            foreach (var axon in _axons)
            {
                var receivedValue = axon.GetValue();
                _value += receivedValue;
            }
        }

        private static float DecayFunction(float value)
        {
            return value * Mathf.Pow(0.5f, Time.fixedDeltaTime);
        }
/*     
        public void Trigger(float value)
        {
            _value += value;
            _value = Mathf.Clamp01(_value);
            //_value = ActivationFunction(_value);
        }
*/
        public void SetValue(float value)
        {
            _value = value;
        }
        private static float ActivationFunction(float value)
        {
            return 1 / (Mathf.Exp(-value) + 1);
        }
    }
}
