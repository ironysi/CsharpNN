using System;
using System.Text;
using NeuralNetworkTesting;

namespace NNConsole
{
    class Program
    {
        static void Main(string[] args)
        {

            NeuralNet net = new NeuralNet();

            double[][] inputs = new double[4][];
            inputs[0] = new double[] { 1, 1 };
            inputs[1] = new double[] { 0, 1 };
            inputs[2] = new double[] { 1, 0 };
            inputs[3] = new double[] { 0, 0 };

            double[][] outputs = new double[4][];
            outputs[0] = new double[] { 0 };
            outputs[1] = new double[] { 1 };
            outputs[2] = new double[] { 1 };
            outputs[3] = new double[] { 0 };


            // initialize with 2 perception neurons 2 hidden layer neurons 1 output neuron
            net.Initialize(1, 2, 2, 1);

            int iterations = 1;
            do
            {
                net.LearningRate = 3;
                net.Train(inputs, outputs, TrainingType.BackPropogation, iterations);

                Console.WriteLine(net.PerceptionLayer[0].Output);
                Console.WriteLine(net.PerceptionLayer[1].Output);
                Console.ReadLine();
                net.PerceptionLayer[0].Output = 0;
                net.PerceptionLayer[1].Output = 0;

                net.Pulse();

                //ll = net.OutputLayer[0].Output;

                net.PerceptionLayer[0].Output = 1;
                net.PerceptionLayer[1].Output = 0;

                net.Pulse();

                //hl = net.OutputLayer[0].Output;

                net.PerceptionLayer[0].Output = 0;
                net.PerceptionLayer[1].Output = 1;

                net.Pulse();

                //lh = net.OutputLayer[0].Output;

                net.PerceptionLayer[0].Output = 1;
                net.PerceptionLayer[1].Output = 1;

                net.Pulse();

                //hh = net.OutputLayer[0].Output;
            }
            // really train this thing well...
            while (true);


            //net.PerceptionLayer[0].Output = low;
            //net.PerceptionLayer[1].Output = low;

            //net.Pulse();

            //ll = net.OutputLayer[0].Output;

            //net.PerceptionLayer[0].Output = high;
            //net.PerceptionLayer[1].Output = low;

            //net.Pulse();

            //hl = net.OutputLayer[0].Output;

            //net.PerceptionLayer[0].Output = low;
            //net.PerceptionLayer[1].Output = high;

            //net.Pulse();

            //lh = net.OutputLayer[0].Output;

            //net.PerceptionLayer[0].Output = high;
            //net.PerceptionLayer[1].Output = high;

            //net.Pulse();

            //hh = net.OutputLayer[0].Output;

            //bld.Remove(0, bld.Length);
            //bld.Append((count * iterations).ToString()).Append(" iterations required for training\n");

            //bld.Append("hh: ").Append(hh.ToString()).Append(" < .5\n");
            //bld.Append("ll: ").Append(ll.ToString()).Append(" < .5\n");

            //bld.Append("hl: ").Append(hl.ToString()).Append(" > .5\n");
            //bld.Append("lh: ").Append(lh.ToString()).Append(" > .5\n");

            //Console.WriteLine(bld.ToString());

            //Console.ReadKey();

        }
    }
}
