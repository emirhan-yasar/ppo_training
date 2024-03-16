using System.Collections.Generic;
using LayerBasedNeuralNetwork.Agent;
using UnityEngine;

namespace LayerBasedNeuralNetwork.VectorMap
{
    public class VectorMap : MonoBehaviour
    {
        [SerializeField] private TheVector vector;
        [SerializeField] private Brain brain;
        [SerializeField] private Transform targetTransform;

        private TheVector[] _vectors;
        private Vector2 _targetPosition;
        private void Awake()
        {
            _vectors = new TheVector[35];
            var i = 0;
            for (var x = -6; x <= 6; x += 2)
            {
                for (var y = -4; y <= 4; y += 2)
                {
                    _vectors[i] = Instantiate(vector, new Vector3(x, y, 0), Quaternion.identity);
                    i++;
                }
            }

            _targetPosition = targetTransform.position;
        }

        private void FixedUpdate()
        {
            foreach (var theVector in _vectors)
            {
                var position = theVector.transform.position;
                var direction = brain.Evaluate(position.x, position.y, _targetPosition.x, _targetPosition.y);
                theVector.SetVector(position, new Vector3(direction[0], direction[1], 0));
            }
        }
    }
}