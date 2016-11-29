using System.Collections.Generic;

namespace NeuralNetworkTesting
{
    public interface INeuronReceptor
    {
        Dictionary<INeuronSignal, NeuralFactor> Input { get; }
    }
}