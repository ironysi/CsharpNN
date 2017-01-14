using System;
using System.Collections.Generic;
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
                Console.WriteLine("5.\tIris Grid Search NN");

                int.TryParse(Console.ReadLine(), out c);

                switch (c)
                {
                    case 1:
                        p.RunIris(0.001, 0.1);
                        break;
                    case 2:
                        p.RunBCancer(0.0000001, 0.005);
                        break;
                    case 3:
                        p.RunANDGate(0.01, 0.1);
                        break;
                    case 4:
                        p.RunWine(0.001, 0.001);
                        break;
                    case 5:
                        p.IrisGridSearch(new int[] { 8, 16/*, 32, 64 */}, new double[] { 0.1, 0.01/*, 0.001*/ });
                        break;
                    case 6:
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
            Console.WriteLine("Hello world, is that You?");
        }

        private void RunWine(double desiredErrorPercentage, double learningRate = 0.1, bool doYouWantToPrint = false)
        {
            string[] categories =
            {
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric","numeric",
                "numeric", "categorical"
            };
            // Output data (quality is 6 numbers)
            Data data = new Data("winequality-red.csv", ';', 11, 6, 0.8, categories);
            NeuralNet net = new NeuralNet(11, 14, 6, learningRate);

            net.RunUntilDesiredError(desiredErrorPercentage, 10, data, doYouWantToPrint);
        }


        private void RunBCancer(double desiredErrorPercentage, double learningRate = 0.05, bool doYouWantToPrint = false)
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

            net.RunUntilDesiredError(desiredErrorPercentage, 5, data, doYouWantToPrint);
        }
        private void RunIris(double desiredErrorPercentage, double learningRate = 0.05, bool doYouWantToPrint = false)
        {
            string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
            Data data = new Data("Iris.txt", ',', 4, 3, 0.8, categories);
            NeuralNet net = new NeuralNet(4, 3, 3, learningRate);

            net.RunUntilDesiredError(desiredErrorPercentage, 50, data, doYouWantToPrint);
            //   net.TestOneRow(Utilities.Concat(data.GetTestingInputs(), data.GetLearningInputs()), data.standardizer);
        }

        private void IrisGridSearch(int[] batchSizes, double[] learningRates)
        {
            for (int i = 0; i < batchSizes.Length; i++)
            {
                for (int k = 0; k < learningRates.Length; k++)
                {
                    Console.WriteLine("\nBatch size: {0}\t\t\t\t\t\t\tLearning rate: {1}", batchSizes[i],
                        learningRates[k]);

                    string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
                    Data data = new Data("Iris.txt", ',', 4, 3, 0.8, categories);
                    NeuralNet net = new NeuralNet(4, 3, 3, learningRates[k]);
                    net.BatchSize = batchSizes[i];
                    net.RunUntilDesiredError(data);
                }
            }

           
        }
        private void RunANDGate(double desiredErrorPercentage, double learningRate = 0.1, bool doYouWantToPrint = false)
        {
            string[] categories = { "numeric", "numeric", "numeric" };
            Data data = new Data("AND.txt", ',', 2, 1, 0.8, categories);
            NeuralNet net = new NeuralNet(2, 2, 1, learningRate);

            net.RunUntilDesiredError(desiredErrorPercentage, 50, data, doYouWantToPrint);
        }

    }
}