using System;

namespace NeuralNetworkTesting
{
    public class Utilities
    {
        public static double DoubleBetween(double min, double max)
        {
            Random rnd = new Random();
            return rnd.NextDouble() * ((double)max - (double)min) + (double)min;
        }

    }
}