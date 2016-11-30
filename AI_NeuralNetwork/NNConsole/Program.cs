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

            BreastsData data = new BreastsData("Breasts.txt", 30, 1, 0.8);
            data.FillData();
            double[][] inputs = data.Inputs.ToRowArrays();
            double[][] outputs = data.Outputs.ToRowArrays();
            // initialize with x perception neurons x hidden layer neurons x output neurons and a learning rate
            double learningRate = 0.1;
            NeuralNet net = new NeuralNet(30, 30, 1, learningRate);

            net.Train(inputs, outputs, 1000);
            //842302,M,
            //17.99,10.38,122.8,
            //1001,0.1184,0.2776,

            //0.3001,0.1471,0.2419,
            //0.07871,1.095,0.9053,
            //8.589,153.4,0.006399,0.04904,

            //0.05373,0.01587,0.03003,
            //0.006193,25.38,17.33,

            //184.6,2019,0.1622,0.6656,
            //0.7119,0.2654,0.4601,0.1189
            //net.InputLayer[0].Output = 17.99;
            //net.InputLayer[1].Output = 10.38;
            //net.InputLayer[2].Output = 122.8;
            //net.InputLayer[4].Output = 1001;
            //net.InputLayer[5].Output = 0.1184;
            //net.InputLayer[6].Output = 0.2776;
            //net.InputLayer[7].Output = 0.3001;
            //net.InputLayer[8].Output = 0.1471;
            //net.InputLayer[9].Output = 0.2419;
            //net.InputLayer[10].Output = 0.07871;
            //net.InputLayer[11].Output = 1.095;
            //net.InputLayer[12].Output = 0.9053;
            //net.InputLayer[13].Output = 8.589;
            //net.InputLayer[14].Output = 153.4;
            //net.InputLayer[15].Output = 0.006399;
            //net.InputLayer[16].Output = 0.04904;
            //net.InputLayer[17].Output = 0.05373;
            //net.InputLayer[18].Output = 0.01587;
            //net.InputLayer[19].Output = 0.03003;
            //net.InputLayer[20].Output = 0.006193;
            //net.InputLayer[21].Output = 25.38;
            //net.InputLayer[22].Output = 17.33;
            //net.InputLayer[23].Output = 184.6;
            //net.InputLayer[24].Output = 2019;
            //net.InputLayer[25].Output = 0.1622;
            //net.InputLayer[26].Output = 0.6656;
            //net.InputLayer[27].Output = 0.7119;
            //net.InputLayer[28].Output = 0.2654;
            //net.InputLayer[29].Output = 0.4601;
            //net.InputLayer[30].Output = 0.1189;
            double[] crapinputs2 =
            {
                18.61, 20.25, 122.1, 1094, 0.0944, 0.1066, 0.149, 0.07731, 0.1697, 0.05699, 0.8529,
                1.849, 5.632, 93.54, 0.01075, 0.02722, 0.05081, 0.01911, 0.02293, 0.004217, 21.31, 27.26, 139.9, 1403,
                0.1338, 0.2117, 0.3446, 0.149, 0.2341, 0.07421
            };
            for (int index = 0; index < crapinputs2.Length; index++)
            {
                net.InputLayer[index].Output = crapinputs2[index];
            }
            net.Pulse();
            Console.WriteLine(net.OutputLayer[0].Output);


            double[] crapinputs =
            {
                8.196, 16.84, 51.71, 201.9, 0.086, 0.05943, 0.01588, 0.005917, 0.1769, 0.06503,
                0.1563, 0.9567, 1.094, 8.205, 0.008968, 0.01646, 0.01588, 0.005917, 0.02574, 0.002582, 8.964, 21.96,
                57.26, 242.2, 0.1297, 0.1357, 0.0688, 0.02564, 0.3105, 0.07409
            };
            for (int index = 0; index < crapinputs.Length; index++)
            {
                net.InputLayer[index].Output = crapinputs[index];
            }

            net.Pulse();
            Console.WriteLine(net.OutputLayer[0].Output);
            //net.Pulse();
            //Console.WriteLine(net.OutputLayer[0].Output);

            //net.InputLayer[0].Output = 5.7;
            //net.InputLayer[1].Output = 2.8;
            //net.InputLayer[2].Output = 4.5;
            //net.InputLayer[3].Output = 1.3;
            //net.Pulse();
            //Console.WriteLine(net.OutputLayer[0].Output);


            Console.ReadLine();
        }
    }
}
