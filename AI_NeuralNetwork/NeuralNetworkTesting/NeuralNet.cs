using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using MyDataSet;

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
        /// <param name="learningRate">Alpha aka. learning rate</param>
        /// <param name="minRange">Lowes value of weights</param>
        /// <param name="maxRange">Highest value of weights</param>
        public NeuralNet(int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount, double learningRate,
            double minRange = 0, double maxRange = 1)
        {
            Random rnd = new Random(1);
            LearningRate = learningRate;

            for (int i = 0; i < inputNeuronCount; i++)
            {
                inputLayer.Add(new Neuron(0));
            }

            for (int i = 0; i < outputNeuronCount; i++)
            {
                outputLayer.Add(new Neuron(rnd.NextDouble()));
            }

            for (int i = 0; i < hiddenNeuronCount; i++)
            {
                hiddenLayer.Add(new Neuron(rnd.NextDouble()));
            }

            /// wire - up input layer to hidden layer
            for (int i = 0; i < hiddenLayer.Count; i++)
                for (int j = 0; j < inputLayer.Count; j++)
                    hiddenLayer[i].Input.Add(inputLayer[j], new NeuralFactor(Utilities.DoubleBetween(minRange, maxRange)));

            //  wire - up output layer to hidden layer
            for (int i = 0; i < outputLayer.Count; i++)
                for (int j = 0; j < hiddenLayer.Count; j++)
                    outputLayer[i].Input.Add(HiddenLayer[j], new NeuralFactor(Utilities.DoubleBetween(minRange, maxRange)));

        }

        /// <summary>
        /// Trains NN until it reaches desired error percentage
        /// </summary>
        /// <param name="desiredError">Desired error percentage</param>
        /// <param name="printErrorEveryXIterations">Indicates how often you print error</param>
        /// <param name="data"></param>
        /// <param name="doYouWantToPrint"></param>
        public void RunUntilDesiredError(double desiredError, int printErrorEveryXIterations, Data data, bool doYouWantToPrint = false)
        {
            double error;
            int i = 1;

            do
            {
                Train(data.GetLearningInputs(), data.GetLearningOutputs(), 10);
                error = TestNetwork(data.GetTestingInputs(), data.GetTestingOutputs(), doYouWantToPrint);
                i++;

                if (i % printErrorEveryXIterations == 0)
                {
                    Console.BackgroundColor = ConsoleColor.Blue;
                    Console.WriteLine(error);
                    Console.ResetColor();
                }

            } while (error > desiredError);

            Console.WriteLine("\nError:{0}\t\t Iterations:{1}", error, i * 10);
            Console.WriteLine("************FINISHED************");
            Console.ReadLine();
        }


        private double TestNetwork(double[][] testingInputs, double[][] testingOutputs, bool doYouWantToPrint)
        {
            double batchError = 0;

            for (int i = 0; i < testingInputs.Length; i++)
            {
                double layerError = 0;

                for (int j = 0; j < testingInputs[0].Length; j++)
                {
                    InputLayer[j].Output = testingInputs[i][j];
                }
                Pulse();


                for (int j = 0; j < testingOutputs[0].Length; j++)
                {
                    layerError += OutputLayer[j].OutputError;
                }
                batchError += layerError / testingOutputs[0].Length;

                if (doYouWantToPrint)
                {
                    PrintResults(testingOutputs, i);
                }

            }
            return batchError / testingOutputs.Length;
        }

        private void PrintResults(double[][] testingOutputs, int indexI)
        {
            Console.Write("DESIRED OUTPUTS:\t");

            for (int j = 0; j < testingOutputs[0].Length; j++)
            {
                Console.Write(testingOutputs[indexI][j] + "\t\t\t");
            }
            Console.Write("\n");

            Console.Write("OUTPUT LAYER: \t");
            for (int j = 0; j < testingOutputs[0].Length; j++)
            {
                Console.Write(OutputLayer[j].Output + "\t");
            }
            Console.Write("\n");
        }

        #region Default code




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
                outputNode.Error = (desiredResults[i] - temp) * Utilities.SigmoidDerivative(temp); //* temp * (1.0F - temp);
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
                    error += (outputNode.Error * outputNode.Input[hiddenNode].Weight) * Utilities.SigmoidDerivative(temp);
                }

                hiddenNode.Error = error;
            }
        }


        public static void PreparePerceptionLayerForPulse(NeuralNet net, double[] input)
        {
            // initialize data
            for (int i = 0; i < net.inputLayer.Count; i++)
                net.inputLayer[i].Output = input[i];
        }

        public static void CalculateAndAppendTransformation(NeuralNet net)
        {
            INeuron hiddenNode;

            // adjust output layer weight change
            for (int j = 0; j < net.outputLayer.Count; j++)
            {
                var outputNode = net.outputLayer[j];

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
                    var inputNode = net.inputLayer[i];
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
        #endregion
    }
}