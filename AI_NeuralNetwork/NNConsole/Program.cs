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
                        p.RunIris(0.0001,0.05);
                        break;
                    case 2:
                        p.RunBCancer(0.001,0.05);
                        break;
                    case 3:
                        p.RunANDGate(0.01,0.1);
                        break;
                    case 4:
                        p.RunWine(0.01,0.005,false);
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

        private void RunWine(double desiredErrorPercentage,double learningRate = 0.1, bool doYouWantToPrint = false)
        {
            string[] categories =
            {
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric","numeric",
                "numeric", "categorical"
            };
            // Output data (quality is 6 numbers)
            Data data = new Data("winequality-red.csv", ';', 11, 6, 0.8, categories);
            NeuralNet net = new NeuralNet(11, 14, 6, learningRate);

            net.RunUntilDesiredError(desiredErrorPercentage, 10, data,doYouWantToPrint);
        }


        private void RunBCancer(double desiredErrorPercentage,double learningRate = 0.05, bool doYouWantToPrint = false)
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
        private void RunIris(double desiredErrorPercentage,double learningRate = 0.05, bool doYouWantToPrint = false)
        {
            string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
            Data data = new Data("Iris.txt", ',', 4, 3, 0.8, categories);
            NeuralNet net = new NeuralNet(4, 3, 3, learningRate,0,1);

            net.RunUntilDesiredError(desiredErrorPercentage, 50, data, doYouWantToPrint);
        }

        private void RunANDGate(double desiredErrorPercentage,double learningRate = 0.1, bool doYouWantToPrint = false)
        {
            string[] categories = { "numeric", "numeric", "numeric" };
            Data data = new Data("AND.txt", ',', 2, 1, 0.8, categories);
            NeuralNet net = new NeuralNet(2, 2, 1, learningRate);

            net.RunUntilDesiredError(desiredErrorPercentage, 50, data, doYouWantToPrint);
        }

    }
}