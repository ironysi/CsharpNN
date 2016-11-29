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
            //double[][] inputs = new double[4][];
            //inputs[0] = new double[] { 1, 1 };
            //inputs[1] = new double[] { 1, 0 };
            //inputs[2] = new double[] { 0, 1 };
            //inputs[3] = new double[] { 0, 0 };

            //double[][] outputs = new double[4][];
            //outputs[0] = new double[] { 1 };
            //outputs[1] = new double[] { 0 };
            //outputs[2] = new double[] { 0 };
            //outputs[3] = new double[] { 0 };

            Data data = new Data("Iris.txt", 4, 1, 0.8);
            data.FillData();
            double[][] inputs = data.Inputs.ToRowArrays();
            double[][] outputs = data.Outputs.ToRowArrays();
            // initialize with x perception neurons x hidden layer neurons x output neurons and a learning rate
            double learningRate = 0.1;
            NeuralNet net = new NeuralNet(4, 4, 1, learningRate);

            net.Train(inputs, outputs, 1000);

            net.InputLayer[0].Output = 5.9;
            net.InputLayer[1].Output = 3;
            net.InputLayer[2].Output = 5.1;
            net.InputLayer[3].Output = 1.8;
            net.Pulse();
            Console.WriteLine(net.OutputLayer[0].Output);

            net.InputLayer[0].Output = 5.7;
            net.InputLayer[1].Output = 2.8;
            net.InputLayer[2].Output = 4.5;
            net.InputLayer[3].Output = 1.3;
            net.Pulse();
            Console.WriteLine(net.OutputLayer[0].Output);

            Console.ReadLine();
        }
    }
}
