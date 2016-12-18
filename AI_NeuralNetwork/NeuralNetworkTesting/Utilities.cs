using System;

namespace NeuralNetworkTesting
{
    public class Utilities
    {
        public static double DoubleBetween(double min, double max)
        {
            Random rnd = new Random(1);
            return rnd.NextDouble() * (max - min) + min;
        }
        //Sigmoid
        public static double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
        public static double SigmoidDerivative(double value)
        {
            return value * (1.0F - value);
        }
        //TANH
        public static double TanH(double value)
        {
            return Math.Tanh(value);
            //return (Math.Exp(value) - Math.Exp(-value)) / (Math.Exp(value) + Math.Exp(-value));
        }

        public static double TanHDerivative(double value)
        {
            return 1.0D - (TanH(value) * TanH(value));
        }
        //BIPOLAR Sigmoid
        public static double BipolarSigmoid(double value)
        {
            return (Math.Exp(2 * value) - 1) / (Math.Exp(2 * value) + 1);
        }


    }
}