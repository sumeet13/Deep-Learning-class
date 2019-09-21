using System;
namespace DataToCSV
{
    public class EpochData
    {
        public int Epoch;
        public double TrainLoss;
        public double ValidationLoss;
        public double ValidationAccuracy;
        public double TestLoss;
        public double TestAccuracy;

        public EpochData(int epoch, double trainLoss, double validationLoss, double validationAccuracy, double testLoss, double testAccuracy)
        {
            Epoch = epoch;
            TrainLoss = trainLoss;
            ValidationLoss = validationLoss;
            ValidationAccuracy = validationAccuracy;
            TestLoss = testLoss;
            TestAccuracy = testAccuracy;
        }
    }
}
