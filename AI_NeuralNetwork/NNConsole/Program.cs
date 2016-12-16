using System;
using System.IO;
using System.Linq;
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
                Console.WriteLine("3.\tAND Gate");

                int.TryParse(Console.ReadLine(), out c);

                switch (c)
                {
                    case 1:
                        p.RunIris();
                        break;
                    case 2:
                        p.RunBCancer();
                        break;
                    case 3:
                        p.RunANDGate();
                        break;
                    case 4:
                        p.RunWine();
                        break;
                    default:
                        Console.WriteLine("Pick a number...");
                        break;
                }

                Console.ReadLine();
                Console.Clear();
            }
        }
        private void RunWine()
        {
            string[] categories =
            {
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric","numeric",
                "numeric", "numeric"
            };

            Data data = new Data("winequality-red.csv", ';', 11, 1, 0.8, categories);
            NeuralNet net = new NeuralNet(11, 11, 1, 0.1);

            RunUntilDesiredError(0.1, 5, data, net);
        }


        private void RunBCancer()
        {
            string[] columnTypes =
            {
                "numeric", "categorical", "numeric", "numeric", "numeric", "numeric", "numeric",
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"
            };
            int[] outputColumns = { 1 };
            int[] ignoredColumns = { 0 };

            Data data = new Data("Breasts.txt", ',', 30, 1, 0.8, columnTypes, outputColumns, ignoredColumns);
            NeuralNet net = new NeuralNet(30, 30, 1, 0.05,-1.0,1.0);

            RunUntilDesiredError(0.001, 5, data, net);
        }
        private void RunIris()
        {
            string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
            Data data = new Data("Iris.txt", ',', 4, 3, 0.8, categories);
            NeuralNet net = new NeuralNet(4, 4, 3, 0.05);

            RunUntilDesiredError(0.02, 50, data, net);
        }

        private void RunANDGate()
        {
            string[] categories = { "numeric", "numeric", "numeric" };
            Data data = new Data("AND.txt", ',', 2, 1, 0.8, categories);
            NeuralNet net = new NeuralNet(2, 2, 1, 0.1);

            RunUntilDesiredError(0.001, 50, data, net);
        }


        private void RunUntilDesiredError(double desiredError, int printErrorEveryXIterations, Data data, NeuralNet net)
        {
            double error;
            int i = 1;

            do
            {
                net.Train(data.LearningInputs.ToRowArrays(), data.LearningOutputs.ToRowArrays(), 10);
                error = TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays());
                i++;

                if (i % printErrorEveryXIterations == 0)
                {
                    Console.BackgroundColor = ConsoleColor.Red;
                    Console.WriteLine(error);
                    Console.ResetColor();
                }

            } while (error > desiredError);

            Console.WriteLine("\nError:{0}\t\t Iterations:{1}", error, i * 10);
            Console.WriteLine("************FINISHED************");
            Console.ReadLine();
        }



        private double TestNetwork(INeuralNet net, double[][] trainingInputs, double[][] trainingOutputs)
        {
            double batchError = 0;



            for (int i = 0; i < trainingInputs.Length; i++)
            {
                double layerError = 0;

                for (int j = 0; j < trainingInputs[0].Length; j++)
                {
                    net.InputLayer[j].Output = trainingInputs[i][j];
                }
                net.Pulse();


                for (int j = 0; j < trainingOutputs[0].Length; j++)
                {
                    layerError += net.OutputLayer[j].OutputError;
                }
                batchError += layerError / trainingOutputs[0].Length;

            //    PrintResults(net,trainingOutputs,i);
            }
            return batchError / trainingOutputs.Length;
        }


        private void PrintResults(INeuralNet net, double[][] trainingOutputs, int indexI)
        {
            Console.Write("DESIRED OUTPUTS:\t");

            for (int j = 0; j < trainingOutputs[0].Length; j++)
            {
                Console.Write(trainingOutputs[indexI][j] + "\t");
            }
            Console.Write("\n");

            Console.Write("OUTPUT LAYER: \t");
            for (int j = 0; j < trainingOutputs[0].Length; j++)
            {
                Console.Write(net.OutputLayer[j].Output + "\t");
            }
            Console.Write("\n");
        }
    }
}