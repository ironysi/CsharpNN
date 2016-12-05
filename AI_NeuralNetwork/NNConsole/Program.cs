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


            while (true)
            {
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
                Console.Clear();
            }
        }

        private void RunBCancer()
        {
            string[] columnTypes = { "numeric", "categorical", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric" };
            int[] outputColumns = { 1 };
            int[] ignoredColumns = { 0 };
            Data data = new Data("Breasts.txt",30,1,0.8,columnTypes,outputColumns,ignoredColumns);

            double[][] learningInputs = data.LearningInputs.ToRowArrays();
            double[][] learningOutputs = data.LearningOutputs.ToRowArrays();
            // initialize with x perception neurons x hidden layer neurons x output neurons and a learning rate
            double learningRate = 0.01;
            NeuralNet net = new NeuralNet(30, 30, 1, learningRate);

            net.Train(learningInputs, learningOutputs, 1000);

            TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays());
        }

        private void RunIris()
        {
            //Data data = new Data("Iris.txt", 4, 1, 0.9);
            string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
            Data data = new Data("Iris.txt", 4,3, 0.8, categories);
            double learningRate = 0.11;
            NeuralNet net = new NeuralNet(4, 4, 3, learningRate);

            double[][] learningInputs = data.LearningInputs.ToRowArrays();
            double[][] learningOutputs = data.LearningOutputs.ToRowArrays();
            net.Train(learningInputs, learningOutputs, 10000);

            TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays());
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
                Console.WriteLine("-----------------------------------------------");
            }
        }
    }
}
