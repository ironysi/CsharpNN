using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace MyDataSet
{
    public class BreastsData
    {
        private Matrix<double> _allDataMatrix;

        private readonly string _fileName;
        private readonly int _inputColumnsCount;

        private List<string> _lines;
        private readonly int _outputColumnsCount;
        private readonly double _percentage;

        public BreastsData()
        {
        }

        public BreastsData(string fileName, int inputColumnsCount, int outputColumnsCount, double percentage)
        {
            _fileName = fileName;
            _inputColumnsCount = inputColumnsCount;
            _outputColumnsCount = outputColumnsCount;
            _percentage = percentage;
        }

        public Matrix<double> Inputs { get; set; }
        public Matrix<double> Outputs { get; set; }

        public void FillData()
        {
            _readLines();
            _shuffleLines();
            double[,] parsed = _parse();
            _allDataMatrix = Matrix<double>.Build.DenseOfArray(parsed);
            Inputs = _allDataMatrix.SubMatrix(0, _allDataMatrix.RowCount, 0, _inputColumnsCount);
            Outputs = _allDataMatrix.SubMatrix(0, _allDataMatrix.RowCount, _inputColumnsCount, _outputColumnsCount);
        }


        public void DivideIntoTestingAndTrainingSet()
        {
            int trainingInputsCount = (int)(_percentage * Inputs.RowCount);
            Matrix<double> trainingInputs = Inputs.SubMatrix(0, trainingInputsCount, 0,
                Inputs.ColumnCount);
            Matrix<double> learningInputs = Inputs.SubMatrix(trainingInputsCount, Inputs.RowCount - trainingInputsCount,
                0, Inputs.ColumnCount);
        }

        private void _readLines()
        {
            _lines = File.ReadLines($"../../../DataSet/Data/{_fileName}").ToList();
        }

        private void _shuffleLines()
        {
            Random rng = new Random(1);
            _lines = _lines.OrderBy(x => rng.Next()).ToList();
        }

        private double[,] _parse()
        {
            string[] firstLine = _lines[0].Split(',');
            double[,] result = new double[_lines.Count,firstLine.Length];

            for (int i = 0; i < _lines.Count; i++)
            {
                string line = _lines[i];
                string[] strings = line.Split(',');

                result[i, 0] = double.Parse(strings[0]);
                result[i, 1] = double.Parse(strings[1]);
                result[i, 2] = double.Parse(strings[2]);
                result[i, 3] = double.Parse(strings[3]);

                switch (strings[4])
                {
                    case "Iris-setosa":
                        result[i, 4] = 0.0001;
                        break;
                    case "Iris-versicolor":
                        result[i, 4] = 0.5000;
                        break;
                    case "Iris-virginica":
                        result[i, 4] = 0.9999;
                        break;
                }
            }
            return result;
        }
    }
}