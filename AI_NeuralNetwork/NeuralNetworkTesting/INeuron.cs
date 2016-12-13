namespace NeuralNetworkTesting
{
    public interface INeuron : INeuronSignal, INeuronReceptor
    {
        void Pulse(INeuralLayer layer);
        void ApplyLearning(INeuralLayer layer, ref double learningRate);
        void InitializeLearning(INeuralLayer layer);

        NeuralFactor Bias { get; set; }

        double Error { get; set; }
        double LastError { get; set; }
        double OutputError { get; set; }
    }
}