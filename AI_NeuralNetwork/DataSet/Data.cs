using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace MyDataSet
{
    public class Data
    {
        private Matrix<double> _allDataMatrix;

        private readonly string _fileName;
        private readonly int _inputColumnsCount;

        private List<string> _lines;
        private readonly int _outputColumnsCount;
        private readonly double _percentage;

        public Data():this("",0,0,0.0)
        {
        }

        public Data(string fileName, int inputColumnsCount, int outputColumnsCount, double percentage)
        {
            _fileName = fileName;
            _inputColumnsCount = inputColumnsCount;
            _outputColumnsCount = outputColumnsCount;
            _percentage = percentage;

            FillData();
        }


        private Matrix<double> _outputs;
        public Matrix<double> LearningOutputs { get; set; }
        public Matrix<double> TrainingOutputs { get; set; }

        private Matrix<double> _inputs;
        public Matrix<double> LearningInputs { get; set; }
        public Matrix<double> TrainingInputs { get; set; }



        public void FillData()
        {
            _readLines();
            _shuffleLines();
            double[,] parsed = _parse();
            _allDataMatrix = Matrix<double>.Build.DenseOfArray(parsed);
            _inputs = _allDataMatrix.SubMatrix(0, _allDataMatrix.RowCount, 0, _inputColumnsCount);
            _outputs = _allDataMatrix.SubMatrix(0, _allDataMatrix.RowCount, _inputColumnsCount, _outputColumnsCount);

            DivideIntoTestingAndTrainingSet();
        }

        public Matrix<double> GetInputs()
        {
            return _inputs;
        }

        public Matrix<double> GetOutputs()
        {
            return _outputs;
        }
             
        public void DivideIntoTestingAndTrainingSet()
        {
            int trainingInputsCount = (int)(_percentage * _inputs.RowCount);

            LearningInputs = _inputs.SubMatrix(0, trainingInputsCount, 0,
                _inputs.ColumnCount);
            TrainingInputs = _inputs.SubMatrix(trainingInputsCount, _inputs.RowCount - trainingInputsCount,
                0, _inputs.ColumnCount);

            LearningOutputs = _outputs.SubMatrix(0, trainingInputsCount, 0,
                _outputs.ColumnCount);
            TrainingOutputs = _outputs.SubMatrix(trainingInputsCount, _outputs.RowCount - trainingInputsCount,
                0, _outputs.ColumnCount);
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
            double[,] result = new double[_lines.Count, firstLine.Length];

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
                        result[i, 4] = 0.5555;
                        break;
                    case "Iris-virginica":
                        result[i, 4] = 0.9999;
                        break;
                }
            }

            //for (int i = 0; i < _lines.Count; i++)
            //{
            //    for (int j = 0; j < 4; j++)
            //    {
            //        result[i, j] = result[i, j]/10;
            //    }
            //}

            return result;
        }
    }
}