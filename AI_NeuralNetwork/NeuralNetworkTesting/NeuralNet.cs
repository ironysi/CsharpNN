using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkTesting
{
    public class NeuralNet : INeuralNet
    {

        private INeuralLayer inputLayer = new NeuralLayer();
        private INeuralLayer outputLayer = new NeuralLayer();
        private INeuralLayer hiddenLayer = new NeuralLayer();

        public INeuralLayer InputLayer => inputLayer;
        public INeuralLayer HiddenLayer => hiddenLayer;
        public INeuralLayer OutputLayer => outputLayer;
        public double LearningRate { get; set; }

        /// <summary>
        /// Includes weight range parameter
        /// </summary>
        /// <param name="inputNeuronCount"></param>
        /// <param name="hiddenNeuronCount"></param>
        /// <param name="outputNeuronCount"></param>
        /// <param name="learningRate"></param>
        /// <param name="minRange">Lowes value of weights</param>
        /// <param name="maxRange">Highest value of weights</param>
        public NeuralNet(int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount, double learningRate, double minRange, double maxRange)
        {
            LearningRate = learningRate;

            for (int i = 0; i < inputNeuronCount; i++)
                inputLayer.Add(new Neuron(0));

            for (int i = 0; i < outputNeuronCount; i++)
                outputLayer.Add(new Neuron(Utilities.DoubleBetween(minRange,maxRange)));

            for (int i = 0; i < hiddenNeuronCount; i++)
                hiddenLayer.Add(new Neuron(Utilities.DoubleBetween(minRange, maxRange)));

            // wire-up input layer to hidden layer
            for (int i = 0; i < hiddenLayer.Count; i++)
                for (int j = 0; j < inputLayer.Count; j++)
                    hiddenLayer[i].Input.Add(inputLayer[j], new NeuralFactor(Utilities.DoubleBetween(minRange, maxRange)));

            // wire-up output layer to hidden layer
            for (int i = 0; i < outputLayer.Count; i++)
                for (int j = 0; j < hiddenLayer.Count; j++)
                    outputLayer[i].Input.Add(HiddenLayer[j], new NeuralFactor(Utilities.DoubleBetween(minRange, maxRange)));
        }

        public NeuralNet(int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount, double learningRate)
        {
            Random rand = new Random(1);
            LearningRate = learningRate;


            for (int i = 0; i < inputNeuronCount; i++)
                inputLayer.Add(new Neuron(0));

            for (int i = 0; i < outputNeuronCount; i++)
                outputLayer.Add(new Neuron(rand.NextDouble()));

            for (int i = 0; i < hiddenNeuronCount; i++)
                hiddenLayer.Add(new Neuron(rand.NextDouble()));

            // wire-up input layer to hidden layer
            for (int i = 0; i < hiddenLayer.Count; i++)
                for (int j = 0; j < inputLayer.Count; j++)
                    hiddenLayer[i].Input.Add(inputLayer[j], new NeuralFactor(rand.NextDouble()));

            // wire-up output layer to hidden layer
            for (int i = 0; i < outputLayer.Count; i++)
                for (int j = 0; j < hiddenLayer.Count; j++)
                    outputLayer[i].Input.Add(HiddenLayer[j], new NeuralFactor(rand.NextDouble()));
        }

        public void Pulse()
        {
            lock (this)
            {
                hiddenLayer.Pulse(this);
                outputLayer.Pulse(this);
            }
        }

        public void ApplyLearning()
        {
            lock (this)
            {
                hiddenLayer.ApplyLearning(this);
                outputLayer.ApplyLearning(this);
            }
        }

        public void InitializeLearning()
        {
            lock (this)
            {
                hiddenLayer.InitializeLearning(this);
                outputLayer.InitializeLearning(this);
            }
        }

        public void Train(double[][] inputs, double[][] outputs, int iterations)
        {
            lock (this)
            {
                for (int i = 0; i < iterations; i++)
                {
                    InitializeLearning(); // set all weight changes to zero

                    for (int j = 0; j < inputs.Length; j++)
                        BackPropogation_TrainingSession(this, inputs[j], outputs[j]);

                    ApplyLearning(); // apply batch of cumlutive weight changes
                }
            }
        }

        #region Methods

        public void PreparePerceptionLayerForPulse(double[] input)
        {
            PreparePerceptionLayerForPulse(this, input);
        }

        private static void CalculateErrors(NeuralNet net, double[] desiredResults)
        {
            double temp;
            INeuron outputNode;

            // Calcualte output error values 
            for (int i = 0; i < net.outputLayer.Count; i++)
            {
                outputNode = net.outputLayer[i];
                temp = outputNode.Output;
                outputNode.OutputError = Math.Abs(desiredResults[i] - temp);
                outputNode.Error = (desiredResults[i] - temp) * SigmoidDerivative(temp); //* temp * (1.0F - temp);
            }

            // calculate hidden layer error values
            for (int i = 0; i < net.hiddenLayer.Count; i++)
            {
                INeuron hiddenNode = net.hiddenLayer[i];
                temp = hiddenNode.Output;

                double error = 0;
                for (int j = 0; j < net.outputLayer.Count; j++)
                {
                    outputNode = net.outputLayer[j];
                    error += (outputNode.Error * outputNode.Input[hiddenNode].Weight) * SigmoidDerivative(temp);// *(1.0F - temp);                   
                }
                
                hiddenNode.Error = error;
            }
        }

        private static double SigmoidDerivative(double value)
        {
            return value * (1.0F - value);
        }

        public static void PreparePerceptionLayerForPulse(NeuralNet net, double[] input)
        {
            // initialize data
            for (int i = 0; i < net.inputLayer.Count; i++)
                net.inputLayer[i].Output = input[i];
        }

        public static void CalculateAndAppendTransformation(NeuralNet net)
        {
            INeuron outputNode, inputNode, hiddenNode;


            // adjust output layer weight change
            for (int j = 0; j < net.outputLayer.Count; j++)
            {
                outputNode = net.outputLayer[j];

                for (int i = 0; i < net.hiddenLayer.Count; i++)
                {
                    hiddenNode = net.hiddenLayer[i];
                    outputNode.Input[hiddenNode].H_Vector += outputNode.Error * hiddenNode.Output;
                }

                outputNode.Bias.H_Vector += outputNode.Error * outputNode.Bias.Weight;
            }

            // adjust hidden layer weight change
            for (int j = 0; j < net.hiddenLayer.Count; j++)
            {
                hiddenNode = net.hiddenLayer[j];

                for (int i = 0; i < net.inputLayer.Count; i++)
                {
                    inputNode = net.inputLayer[i];
                    hiddenNode.Input[inputNode].H_Vector += hiddenNode.Error * inputNode.Output;
                }

                hiddenNode.Bias.H_Vector += hiddenNode.Error * hiddenNode.Bias.Weight;
            }

        }

        public static void BackPropogation_TrainingSession(NeuralNet net, double[] input, double[] desiredResult)
        {
            PreparePerceptionLayerForPulse(net, input);
            net.Pulse();
            CalculateErrors(net, desiredResult);
            CalculateAndAppendTransformation(net);
        }

        #endregion Private Static Utility Methods -------------------------------------------
    }
}
