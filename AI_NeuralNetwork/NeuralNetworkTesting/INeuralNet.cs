namespace NeuralNetworkTesting
{
    public interface INeuralNet
    {
        INeuralLayer PerceptionLayer { get; }
        INeuralLayer HiddenLayer { get; }
        INeuralLayer OutputLayer { get; }

        double LearningRate { get; set; }

        void Pulse();
        void ApplyLearning();
        void InitializeLearning();
    }
}