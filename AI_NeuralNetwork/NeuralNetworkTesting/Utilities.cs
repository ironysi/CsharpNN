using System;
using MathNet.Numerics.Statistics;

namespace NeuralNetworkTesting
{
    public class Utilities
    {


        //ReLU
        public static double ReLU(double value)
        {
            return Math.Max(0, value);
        }
        public static double ReLUDerivative(double value)
        {
            return value > 0 ? 1 : 0;
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

        /// <summary>
        /// Random double between two numbers
        /// </summary>
        public static double DoubleBetween(double min, double max)
        {
            Random rnd = new Random(1);
            return rnd.NextDouble() * (max - min) + min;
        }

        public static double ZScore(double x, double[][] allValues)
        {
            double a = allValues[0].Mean();
            double b = allValues[0].StandardDeviation();

            double y = (x - a) / b;
            return y;
        }

        public static double[][] Concat(double[][] array1, double[][] array2)
        {
            double[][] result = new double[array1.Length + array2.Length][];

            array1.CopyTo(result, 0);
            array2.CopyTo(result, array1.Length);

            return result;
        }

    }
}