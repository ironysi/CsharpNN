namespace NeuralNetworkTesting
{
    public interface INeuralNet
    {
        INeuralLayer InputLayer { get; }
        INeuralLayer HiddenLayer { get; }
        INeuralLayer OutputLayer { get; }

        double LearningRate { get; set; }

        void Pulse();
        void ApplyLearning();
        void InitializeLearning();
    }
}