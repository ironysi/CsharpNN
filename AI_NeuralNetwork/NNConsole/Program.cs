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
                        p.RunANDGate();
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

            RunUntilDesiredError(0.1, 50, data, net);
        }

        private void RunANDGate()
        {
            string[] categories = { "numeric", "numeric", "categorical"};
            Data data = new Data("AND.txt", ',', 2, 2, 0.8, categories);
            NeuralNet net = new NeuralNet(2, 2, 2, 0.1);

            RunUntilDesiredError(0.1, 50, data, net);
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

                //Console.BackgroundColor = ConsoleColor.Red;
                //Console.WriteLine("***************--------------------END OF 10 ITERATION--------------------***************");
                //Console.ResetColor();
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
                error[i] = Math.Abs(trainingOutputs[i][0] - net.OutputLayer[0].Output);
                //IRIS
                //Console.WriteLine("INPUT LAYER: {0}\t{1}\t{2}\t{3}", net.InputLayer[0].Output, net.InputLayer[1].Output, net.InputLayer[2].Output, net.InputLayer[3].Output);
                //Console.WriteLine("SUPPOSED TO BE: {0}\t{1}\t{2}", trainingOutputs[i][0], trainingOutputs[i][1], trainingOutputs[i][2]);
                //Console.WriteLine("OUTPUT LAYER: {0}\t{1}\t{2}\tERROR: {3}\t{4}\t{5}",net.OutputLayer[0].Output, net.OutputLayer[1].Output, net.OutputLayer[2].Output,net.OutputLayer[0].Error, net.OutputLayer[1].Error,net.OutputLayer[2].Error);

                //AND
                Console.WriteLine("INPUT LAYER: {0}\t{1}", net.InputLayer[0].Output, net.InputLayer[1].Output);
                Console.WriteLine("OUTPUT LAYER: {0}\t{1}\tERROR: {2}\t{3}",net.OutputLayer[0].Output, net.OutputLayer[1].Output,net.OutputLayer[0].Error, net.OutputLayer[1].Error);


                //if (error[i] > 0.3)
                //{

                //    for (int k = 0; k < trainingOutputs[i].Length; k++)
                //    {
                //        Console.BackgroundColor = ConsoleColor.Blue;
                //        Console.WriteLine(trainingOutputs[i][k] + "\t\t\t" + net.OutputLayer[k].Output + "\t" + net.OutputLayer[k].Error);
                //        Console.ResetColor();
                //    }

                //}
                //else
                //{
                //    for (int k = 0; k < trainingOutputs[i].Length; k++)
                //    {
                //        Console.WriteLine(trainingOutputs[i][k] + "\t\t\t" + net.OutputLayer[k].Output + "\t" + net.OutputLayer[k].Error);
                //        Console.WriteLine(trainingOutputs[i][k] + "\t\t\t" + net.OutputLayer[k].Output + "\t" + error[i]);
                //    }
                //}
                Console.ReadLine();
            }

            double bad = error.Where(x => x > 0.3).Count();
            double percentage = (bad / error.Length) * 100;

            return percentage;
        }
    }
}
