#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <string>
#include <fstream>
#include <utility>
#include <random>
#include <set>

typedef std::vector<std::vector<double>> Matrix;

class KNeighborsClassifier {
private:
    size_t KNeighbors;
    Matrix coordinateMatrix;
    std::vector<int> labels;
    void CheckInputFit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) const;
    void CheckInputPredict(const Matrix& matrixToPredict) const;
    int PredictSingleCoordinate(const std::vector<double>& vectorToPredict) const;
public:
    KNeighborsClassifier(size_t);
    void Fit(const Matrix&, const std::vector<int>&);
    std::vector<int> Predict(const Matrix& matrixToPredict) const;
};

void CheckMatrixRectangular(const Matrix& matrixChecked);
double MeasureDistanceCoordinates(const std::vector<double>& firstCoordinate, const std::vector<double>& secondCoordinate);

std::vector<int> CreateLabels(const std::string documentName);
Matrix CreateMatrix(const std::string documentName);
void Split(const std::string& line, char delimiter, Matrix* matrixObject);
double Accuracy(std::vector<int> first, std::vector<int> second);

void TrainTestSplit(const Matrix& inputMatrix, const std::vector<int>& inputLabels,
                            Matrix* trainMatrix, std::vector<int>* trainLabels,
                            Matrix* testMatrix, std::vector<int>* testLabels, double ratioAllocateToTrain);