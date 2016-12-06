﻿using System;
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
                Console.WriteLine("3.\tWine");

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
            string[] columnTypes =
            {
                "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                "numeric", "numeric", "numeric"
            };

            Data data = new Data("winequality-red.csv",';', 11, 1, 0.8, columnTypes);
            NeuralNet net = new NeuralNet(11, 11, 1, 0.01);

            RunUntilDesiredError(0.01, 5, data, net);
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
           
            Data data = new Data("Breasts.txt",',', 30, 1, 0.8, columnTypes, outputColumns, ignoredColumns);
            NeuralNet net = new NeuralNet(30, 30, 1, 0.005);

            RunUntilDesiredError(0.01, 5, data, net);
        }
        private void RunIris()
        {
            string[] categories = { "numeric", "numeric", "numeric", "numeric", "categorical" };
            Data data = new Data("Iris.txt",',', 4, 3, 0.8, categories);
            NeuralNet net = new NeuralNet(4, 4, 3, 0.01);

            RunUntilDesiredError(0.5, 50, data, net);
        }
        private void RunUntilDesiredError(double desiredError, int printErrorEveryXIterations, Data data, NeuralNet net)
        {
            double error;
            int i = 1;

            do
            {
                net.Train(data.LearningInputs.ToRowArrays(), data.LearningInputs.ToRowArrays(), 10);
                error = TestNetwork(net, data.TrainingInputs.ToRowArrays(), data.TrainingOutputs.ToRowArrays());
                i++;

                if (i % printErrorEveryXIterations == 0)
                {
                    Console.WriteLine(error);
                }

                Console.BackgroundColor = ConsoleColor.Red;
                Console.WriteLine("***************--------------------END OF 10 ITERATION--------------------***************");
                Console.ResetColor();
            } while (error > desiredError);

            Console.WriteLine("\nError:{0}\t\t Iterations:{1}", error, i * 10);
            Console.WriteLine("************FINISHED************");
            Console.ReadLine();
        }



        private double TestNetwork(INeuralNet net, double[][] trainingInputs, double[][] trainingOutputs)
        {
            double[] error = new double[trainingInputs.Length];


            for (int i = 0; i < trainingInputs.Length; i++)
            {
                for (int j = 0; j < trainingInputs[0].Length; j++)
                {
                    net.InputLayer[j].Output = trainingInputs[i][j];
                }

                net.Pulse();
                error[i] = trainingOutputs[i][0] - net.OutputLayer[0].Output;

                if (error[i] < -0.3)
                {

                    for (int k = 0; k < trainingOutputs[i].Length; k++)
                    {
                        Console.BackgroundColor = ConsoleColor.Blue;
                        Console.WriteLine(trainingOutputs[i][k] + "\t\t\t" + net.OutputLayer[0].Output + "\t" + error[i]);
                        Console.ResetColor();
                    }

                }
                else
                {
                    for (int k = 0; k < trainingInputs[i].Length; k++)
                    {
                        Console.WriteLine(trainingOutputs[i][0] + "\t\t\t" + net.OutputLayer[0].Output + "\t" + error[i]);
                    }
                }
            }

            double bad = error.Where(x => x < -0.3).Count();
            double percentage = (bad / error.Length) * 100;

            return percentage;
        }
    }
}
