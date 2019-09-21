data = read.csv("data_model8.csv")

maxLoss = max(c(max(data$TestLoss, data$TrainLoss)))
minLoss = min(c(min(data$TestLoss, data$TrainLoss)))

plot(TestLoss ~ Epoch, data, col="red", lyt=1, ylim=c(minLoss,maxLoss), ylab="Loss")
lines(data$Epoch, data$TestLoss, col="red")

points(TrainLoss ~ Epoch, data, col="green", lyt=2)
lines(data$Epoch, data$TrainLoss, col="green")

