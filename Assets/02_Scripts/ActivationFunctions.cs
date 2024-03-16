using UnityEngine;

public static class ActivationFunctions
{
    public static float Sigmoid(float value)
    {
        return 1 / (1 + Mathf.Exp(-value));
    }

    public static float Swish(float value)
    {
        return value / (1 + Mathf.Exp(-value));
    }

    public static float TanH(float value)
    {
        var a = Mathf.Exp(value);
        var b = Mathf.Exp(-value);
        return (a - b) / (a + b);
    }

    public static float LinearClip(float value)
    {
        // Deterministic:
        return Mathf.Clamp(value, -3f, 3f) / 3f;
    }
}