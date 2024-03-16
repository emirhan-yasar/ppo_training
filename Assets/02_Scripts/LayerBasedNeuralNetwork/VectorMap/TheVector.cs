using UnityEngine;

namespace LayerBasedNeuralNetwork.VectorMap
{
    [RequireComponent(typeof(LineRenderer))]
    public class TheVector : MonoBehaviour
    {
        [SerializeField] private Transform tipTransform;

        private LineRenderer _lineRenderer;
        private void Awake()
        {
            _lineRenderer = GetComponent<LineRenderer>();
            _lineRenderer.positionCount = 2;
        }

        public void SetVector(Vector3 position, Vector3 direction)
        {
            transform.position = position;
            _lineRenderer.SetPosition(0, position);
            _lineRenderer.SetPosition(1, position + direction);
            tipTransform.position = position + direction;
            tipTransform.right = direction;
        }
        
    }
}