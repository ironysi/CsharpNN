using System;
using System.Text;
using MyDataSet;
using NeuralNetworkTesting;

namespace NNConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Program p = new Program();

            int c;

            Console.WriteLine(@"******** Welcome! ********");
            Console.WriteLine("Please select which Neural Network you want to run.\n");
            Console.WriteLine("1.\tIris NN");
            Console.WriteLine("2.\tBreast Cancer NN");

            int.TryParse(Console.ReadLine(), out c);

            switch (c)
            {
                case 1:
                    p.RunIris();
                    break;
                case 2:
                    p.RunBCancer();
                    break;
                default:
                    Console.WriteLine("Pick a number...");
                    break;
            }

            Console.ReadLine();
        }

        private void RunBCancer()
        {
            BreastsData data = new BreastsData("Breasts.txt", 30, 1, 0.8);

            double[][] learningInputs = data.LearningInputs.ToRowArrays();
            double[][] learningOutputs = data.LearningOutputs.ToRowArrays();
            // initialize with x perception neurons x hidden layer neurons x output neurons and a learning rate
            double learningRate = 0.1;
            NeuralNet net = new NeuralNet(30, 30, 1, learningRate);

            net.Train(learningInputs, learningOutputs, 1000);

            TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays());

            /////////////////////////////////////////

            //// M --> 0
            //double[] crapinputs2 =
            //{
            //    18.61, 20.25, 122.1, 1094, 0.0944, 0.1066, 0.149, 0.07731, 0.1697, 0.05699, 0.8529,
            //    1.849, 5.632, 93.54, 0.01075, 0.02722, 0.05081, 0.01911, 0.02293, 0.004217, 21.31, 27.26, 139.9, 1403,
            //    0.1338, 0.2117, 0.3446, 0.149, 0.2341, 0.07421
            //};

            //for (int index = 0; index < crapinputs2.Length; index++)
            //{
            //    net.InputLayer[index].Output = crapinputs2[index];
            //}
            //net.Pulse();
            //Console.WriteLine(net.OutputLayer[0].Output);

            //// B --> 1 
            //double[] crapinputs =
            //{
            //    8.196, 16.84, 51.71, 201.9, 0.086, 0.05943, 0.01588, 0.005917, 0.1769, 0.06503,
            //    0.1563, 0.9567, 1.094, 8.205, 0.008968, 0.01646, 0.01588, 0.005917, 0.02574, 0.002582, 8.964, 21.96,
            //    57.26, 242.2, 0.1297, 0.1357, 0.0688, 0.02564, 0.3105, 0.07409
            //};
            //for (int index = 0; index < crapinputs.Length; index++)
            //{
            //    net.InputLayer[index].Output = crapinputs[index];
            //}

            //net.Pulse();
            //Console.WriteLine(net.OutputLayer[0].Output);

            Console.ReadLine();
        }

        private void RunIris()
        {
            Data data = new Data("Iris.txt", 4, 1, 0.8);
            double learningRate = 0.05;
            // initialize with x perception neurons x hidden layer neurons x output neurons and a learning rate
            NeuralNet net = new NeuralNet(4, 4, 1, learningRate);

            double[][] learningInputs = data.LearningInputs.ToRowArrays();
            double[][] learningOutputs = data.LearningOutputs.ToRowArrays();
            net.Train(learningInputs, learningOutputs, 1500);

            TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays());

            Console.ReadLine();
        }

        private void TestNetwork(INeuralNet net, double[][] trainingInputs, double[][] trainingOutputs)
        {
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                for (int j = 0; j < trainingInputs[0].Length; j++)
                {
                    net.InputLayer[j].Output = trainingInputs[i][j];
                }

                net.Pulse();

                Console.WriteLine(trainingOutputs[i][0] + "\t" + net.OutputLayer[0].Output);
            }
        }
    }
}
