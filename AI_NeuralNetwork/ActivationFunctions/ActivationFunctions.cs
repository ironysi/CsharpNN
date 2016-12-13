using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ActivationFunctions
{
    public static class ActFunc
    {
        public static double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
        public static double BipolarSigmoid(double value)
        {
            return 2 / (1 + Math.Exp(-2 * value)) - 1;
        }
    }
}
