using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkTesting
{
    public class NeuralLayer : INeuralLayer
    {
        #region Constructor

        public NeuralLayer()
        {
            m_neurons = new List<INeuron>();
        }

        #endregion

        #region Member Variables

        private List<INeuron> m_neurons;

        #endregion

        #region IList<INeuron> Members

        public int IndexOf(INeuron item)
        {
            return m_neurons.IndexOf(item);
        }

        public void Insert(int index, INeuron item)
        {
            m_neurons.Insert(index, item);
        }

        public void RemoveAt(int index)
        {
            m_neurons.RemoveAt(index);
        }

        public INeuron this[int index]
        {
            get { return m_neurons[index]; }
            set { m_neurons[index] = value; }
        }

        #endregion

        #region ICollection<INeuron> Members

        public void Add(INeuron item)
        {
            m_neurons.Add(item);
        }

        public void Clear()
        {
            m_neurons.Clear();
        }

        public bool Contains(INeuron item)
        {
            return m_neurons.Contains(item);
        }

        public void CopyTo(INeuron[] array, int arrayIndex)
        {
            m_neurons.CopyTo(array, arrayIndex);
        }

        public int Count
        {
            get { return m_neurons.Count; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(INeuron item)
        {
            return m_neurons.Remove(item);
        }

        #endregion

        public IEnumerator<INeuron> GetEnumerator()
        {
            return m_neurons.GetEnumerator();
        }



        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }


        #region INeuralLayer Members

        public void Pulse(INeuralNet net)
        {
            foreach (INeuron n in m_neurons)
                n.Pulse(this);
        }

        public void ApplyLearning(INeuralNet net)
        {
            double learningRate = net.LearningRate;

            foreach (INeuron n in m_neurons)
                n.ApplyLearning(this, ref learningRate);
        }

        public void InitializeLearning(INeuralNet net)
        {
            foreach (INeuron n in m_neurons)
                n.InitializeLearning(this);
        }

        #endregion
    }
}
