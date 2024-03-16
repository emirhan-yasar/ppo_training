using UnityEngine;

namespace PPONeuralNetwork.PPO
{
    public class Experience
    {
        public float[] State { get; set; }
        
        public float[] Actions { get; set; }
        public float Reward { get; set; }
        public float[] NextState { get; set; }
        
        public bool Done { get; set; }

        public Experience(float[] state, float[] actions, float reward, float[] nextState, bool done)
        {
            State = state;
            Actions = actions;
            Reward = reward;
            NextState = nextState;
            Done = done;
        }
    }
}