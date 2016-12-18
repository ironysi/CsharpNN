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
                Console.WriteLine("4.\tWine NN");

                int.TryParse(Console.ReadLine(), out c);

                switch (c)
                {
                    case 1:
                        p.RunIris(0.05);
                        break;
                    case 2:
                        p.RunBCancer(0.05, true);
                        break;
                    case 3:
                        p.RunANDGate(0.1);
                        break;
                    case 4:
                        p.RunWine(0.005,false);
                        break;
                    case 5:
                        p.Test();
                        break;
                    default:
                        Console.WriteLine("Pick a number...");
                        break;
                }

                Console.ReadLine();
                Console.Clear();
            }
        }

        private void Test()
        {
            Console.WriteLine("rnd:\t" + Math.Pow(10,2));
            Console.WriteLine("util:\t" + Utilities.DoubleBetween(0, 1));
        }

        private void RunWine(double learningRate = 0.1, bool doYouWantToPrint = false)
        {
            string[] categories =
            {
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric","numeric",
                "numeric", "categorical"
            };
            // Output data (quality is 6 numbers)
            Data data = new Data("winequality-red.csv", ';', 11, 6, 0.8, categories);
            NeuralNet net = new NeuralNet(11, 14, 6, learningRate);

            RunUntilDesiredError(0.1, 10, data, net,doYouWantToPrint);
        }


        private void RunBCancer(double learningRate = 0.05, bool doYouWantToPrint = false)
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
            NeuralNet net = new NeuralNet(30, 30, 1, learningRate, -1.0, 1.0);

            RunUntilDesiredError(0.000001, 5, data, net, doYouWantToPrint);
        }
        private void RunIris(double learningRate = 0.05, bool doYouWantToPrint = false)
        {
            string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
            Data data = new Data("Iris.txt", ',', 4, 3, 0.8, categories);
            NeuralNet net = new NeuralNet(4, 3, 3, learningRate);

            RunUntilDesiredError(0.02, 50, data, net, doYouWantToPrint);
        }

        private void RunANDGate(double learningRate = 0.1, bool doYouWantToPrint = false)
        {
            string[] categories = { "numeric", "numeric", "numeric" };
            Data data = new Data("AND.txt", ',', 2, 1, 0.8, categories);
            NeuralNet net = new NeuralNet(2, 2, 1, learningRate);

            RunUntilDesiredError(0.01, 50, data, net, doYouWantToPrint);
        }


        private void RunUntilDesiredError(double desiredError, int printErrorEveryXIterations, Data data, NeuralNet net, bool doYouWantToPrint = false)
        {
            double error;
            int i = 1;

            do
            {
                net.Train(data.LearningInputs.ToRowArrays(), data.LearningOutputs.ToRowArrays(), 10);
                error = TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays(), doYouWantToPrint);
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



        private double TestNetwork(INeuralNet net, double[][] trainingInputs, double[][] trainingOutputs, bool doYouWantToPrint)
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

                if (doYouWantToPrint)
                {
                    PrintResults(net, trainingOutputs, i);
                }

            }
            return batchError / trainingOutputs.Length;
        }


        private void PrintResults(INeuralNet net, double[][] trainingOutputs, int indexI)
        {
            Console.Write("DESIRED OUTPUTS:\t");

            for (int j = 0; j < trainingOutputs[0].Length; j++)
            {
                Console.Write(trainingOutputs[indexI][j] + "\t\t\t");
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