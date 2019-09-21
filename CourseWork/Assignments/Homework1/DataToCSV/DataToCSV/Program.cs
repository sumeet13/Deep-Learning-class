using System;
using System.Collections.Generic;
using System.IO;

namespace DataToCSV
{
    class MainClass
    {
        const int totalValidationCount = 4800;
        const int totalTestCount = 12000;

        public static List<EpochData> readSlurmFile(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath);

            bool isInEpochs = false;
            List<EpochData> Epochs = new List<EpochData>();
            for (int i = 0; i < lines.Length;)
            {
                if (isInEpochs == false)
                {
                    if (lines[i].StartsWith("Epoch: ", StringComparison.CurrentCulture) == false)
                    {
                        i++;
                        continue;
                    }

                    isInEpochs = true;
                }

                int epochNum = int.Parse(lines[i].Split(' ')[1]);
                i++;
                double trainCrossEntropy = double.Parse(lines[i].Split(' ')[3]);
                i++;
                double validationCrossEntropy = double.Parse(lines[i].Split(' ')[3]);
                i++;

                int totalValidationCorrect = 0;
                i++;
                for (int rows = 0; rows < 10; rows++)
                {
                    var row = lines[i];
                    i++;
                    var trimmedRow = row.Trim(' ', '[', ']');

                    var values = trimmedRow.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    totalValidationCorrect += int.Parse(values[rows]);
                }

                double percentageValidationCorrect = 1.0*totalValidationCorrect / totalValidationCount;


                if (lines[i].StartsWith("STOPPING EPOCH", StringComparison.CurrentCulture))
                {
                    break;
                }

                double testCrossEntropy = double.Parse(lines[i].Split(' ')[3]);
                i++;

                int totalTestCorrect = 0;
                i++;
                for (int rows = 0; rows < 10; rows++)
                {
                    var row = lines[i];
                    i++;
                    var trimmedRow = row.Trim(' ', '[', ']');

                    var values = trimmedRow.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    totalTestCorrect += int.Parse(values[rows]);
                }

                double percentageTestCorrect = 1.0*totalTestCorrect / totalTestCount;

                Epochs.Add(new EpochData(epochNum, trainCrossEntropy, validationCrossEntropy, percentageValidationCorrect, testCrossEntropy, percentageTestCorrect));
            }

            return Epochs;
        }

        public static void Main(string[] args)
        {
            string[] files = Directory.GetFiles("Data");
            foreach(var file in files)
            {
                var name = file.Split('/')[file.Split('/').Length - 1].Split('.')[0];
                List<EpochData> epochs = readSlurmFile(file);

                File.WriteAllLines("data_" + name + ".csv", new string[] { "Epoch,TrainLoss,ValidationLoss,ValidationAccuracy,TestLoss,TestAccuracy" });
                string[] csvLines = new string[epochs.Count];
                for(int i = 0; i < epochs.Count; i++)
                {
                    EpochData epoch = epochs[i];
                    csvLines[i] = String.Format("{0},{1},{2},{3},{4},{5}", epoch.Epoch, epoch.TrainLoss, epoch.ValidationLoss, epoch.ValidationAccuracy, epoch.TestLoss, epoch.TestAccuracy);
                }
                File.AppendAllLines("data_" + name + ".csv", csvLines);
            }
        }
    }
}
