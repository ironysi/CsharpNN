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
            double[,] result = new double[_lines.Count, firstLine.Length-1];

            for (int i = 0; i < _lines.Count; i++)
            {
                string line = _lines[i];
                string[] strings = line.Split(',');

                for (int j = 0; j < 30; j++)
                {
                    result[i, j] = double.Parse(strings[j+2]);
                }

                switch (strings[1])
                {
                    case "M":
                        result[i, 30] = 0.0001;
                        break;
                    case "B":
                        result[i, 30] = 0.999;
                        break;
                }
            }
            return result;
        }
    }
}