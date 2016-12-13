using System;
using System.Collections.Generic;
using ActivationFunctions;

namespace NeuralNetworkTesting
{
    public class Neuron : INeuron
    {

        public Neuron(double bias, ActFunction actFunction)
        {
            m_bias = new NeuralFactor(bias);
            m_error = 0;
            m_input = new Dictionary<INeuronSignal, NeuralFactor>();
            _activationFunction = actFunction;
        }

        public Neuron(double bias) : this(bias, ActFunc.Sigmoid)
        {
        }



        private Dictionary<INeuronSignal, NeuralFactor> m_input;
        double m_output, m_error, m_lastError;
        NeuralFactor m_bias;
<<<<<<< HEAD

=======
        public delegate double ActFunction(double value);
        private ActFunction _activationFunction;
        #endregion

        #region INeuronSignal Members
>>>>>>> 7884e21d85bf035716557c03bc3a7d200827dcb3

        public double Output
        {
            get { return m_output; }
            set { m_output = value; }
        }



        public Dictionary<INeuronSignal, NeuralFactor> Input
        {
            get { return m_input; }
        }


        public void Pulse(INeuralLayer layer)
        {
            lock (this)
            {
                m_output = 0;

                foreach (KeyValuePair<INeuronSignal, NeuralFactor> item in m_input)
                {
                    m_output += item.Key.Output * item.Value.Weight;
                }
                m_output += m_bias.Weight;

<<<<<<< HEAD

                //ACT Function is HERE 

                    m_output = Sigmoid(m_output);
              //  m_output = TanH(m_output);
                //     m_output = BipolarSigmoid(m_output);
=======
                m_output = _activationFunction(m_output);
>>>>>>> 7884e21d85bf035716557c03bc3a7d200827dcb3
            }
        }

        public NeuralFactor Bias
        {
            get { return m_bias; }
            set { m_bias = value; }
        }


        public double OutputError { get; set;}
        public double Error
        {
            get { return m_error; }
            set
            {
                m_lastError = m_error;
                m_error = value;
            }
        }

        public void ApplyLearning(INeuralLayer layer, ref double learningRate)
        {
            foreach (KeyValuePair<INeuronSignal, NeuralFactor> m in m_input)
                m.Value.ApplyWeightChange(ref learningRate);

            m_bias.ApplyWeightChange(ref learningRate);
        }

        public void InitializeLearning(INeuralLayer layer)
        {
            foreach (KeyValuePair<INeuronSignal, NeuralFactor> m in m_input)
                m.Value.ResetWeightChange();

            m_bias.ResetWeightChange();
        }

        public double LastError
        {
            get { return m_lastError; }
            set { m_lastError = value; }
        }

<<<<<<< HEAD

        public static double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }


        /// <summary>
        /// !!! NOT SURE ABOUT THIS !!!
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double BipolarSigmoid(double value)
        {
            return 2 / (1 + Math.Exp(-2 * value)) - 1;
        }

        public static double TanH(double value)
        {
            return Math.Tanh(value);
        }
=======
        #endregion
>>>>>>> 7884e21d85bf035716557c03bc3a7d200827dcb3

    }
}