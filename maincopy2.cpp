#include "kNN.h"

typedef std::vector<std::vector<double>> Matrix;

int main(int argc, char const *argv[]) {
    Matrix inputMatrix = CreateMatrix("data.csv");
    std::vector<int> inputLabels = CreateLabels("target.csv");

    Matrix trainMatrix;
    std::vector<int> trainLabels;
    
    Matrix testMatrix;
    std::vector<int> testLabels;

    double ratioToTrain = 0.1;

    TrainTestSplit(inputMatrix, inputLabels, &trainMatrix, &trainLabels, &testMatrix, &testLabels, ratioToTrain);

    KNeighborsClassifier classifierKnnSizeTen(3);
    classifierKnnSizeTen.Fit(trainMatrix, trainLabels);

    std::vector<int> predictOutput = classifierKnnSizeTen.Predict(testMatrix);

    double accuracyRate = Accuracy(testLabels, predictOutput);

    std::cout << accuracyRate << "\n";

    return 0;
}