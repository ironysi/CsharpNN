namespace NeuralNetworkTesting
{
    public class NeuralFactor
    {
        #region Constructors

        public NeuralFactor(double weight)
        {
            m_weight = weight;
            m_lastDelta = m_delta = 0;
        }

        #endregion

        #region Member Variables

        private double m_weight, m_lastDelta, m_delta;

        #endregion

        #region Properties

        public double Weight
        {
            get { return m_weight; }
            set { m_weight = value; }
        }

        public double H_Vector
        {
            get { return m_delta; }
            set { m_delta = value; }
        }

        public double Last_H_Vector
        {
            get { return m_lastDelta; }
            set { m_lastDelta = value; }
        }

        #endregion

        #region Methods

        public void ApplyWeightChange(ref double learningRate)
        {
            m_lastDelta = m_delta;
            m_weight += m_delta * learningRate;
        }

        public void ResetWeightChange()
        {
            m_lastDelta = m_delta = 0;
        }

        #endregion
    }
}