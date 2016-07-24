#include "kNN.h"

static const int thisIsZero = 0;

KNeighborsClassifier::KNeighborsClassifier(size_t kneighbors): KNeighbors(kneighbors) {}

void KNeighborsClassifier::CheckInputFit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) const {
    if (neighborCoordinates.size() != neighborLabels.size()) {
        std::cerr << "bad input\n" << "neighborCoordinates.size() == " << neighborCoordinates.size() 
                    << " neighborLabels.size() == " << neighborLabels.size() << "\n";
        exit(1);
    }
    if (KNeighbors > neighborLabels.size()) {
        std::cerr << "bad input\n" << "KNeighbors == " << KNeighbors
                    << " coordinateMatrix.size() == " << neighborLabels.size() << "\n";
        exit(1);
    }
    CheckMatrixRectangular(neighborCoordinates);
}

void CheckMatrixRectangular(const Matrix& matrixChecked) {
    for (size_t i = 0; i < matrixChecked.size() - 1; ++i) {
        if (matrixChecked[i].size() != matrixChecked[i + 1].size()) {
            std::cerr << "Matrix is not rectangular\n" << "matrixChecked[i].size() == " << matrixChecked[i].size()
                        << " matrixChecked[i].size() == " << matrixChecked[i + 1].size() << "\n";
            exit(1);
        }
    }
}

void KNeighborsClassifier::Fit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) {
    CheckInputFit(neighborCoordinates, neighborLabels);
    coordinateMatrix = neighborCoordinates;
    labels = neighborLabels;
}

double MeasureDistanceCoordinates(const std::vector<double>& firstCoordinate, const std::vector<double>& secondCoordinate) {
    double distanceSquared = 0;
        for (int j = 0; j < firstCoordinate.size(); ++j) {
            distanceSquared += pow(firstCoordinate[j] - secondCoordinate[j], 2);
        }
    return sqrt(distanceSquared);
}

void KNeighborsClassifier::CheckInputPredict(const Matrix& matrixToPredict) const {
    CheckMatrixRectangular(matrixToPredict);
    if (coordinateMatrix[thisIsZero].size() != matrixToPredict[thisIsZero].size()) {
        std::cerr << "different dimensions\n" << "coordinateMatrix[thisIsZero].size() == " << coordinateMatrix[thisIsZero].size()
                    << " matrixToPredict[thisIsZero].size() == " << matrixToPredict[thisIsZero].size();
        exit(1);
    }
}

int KNeighborsClassifier::PredictSingleCoordinate(const std::vector<double>& vectorToPredict) const {
    std::priority_queue<std::pair<double, size_t>> queueDistanceLabelPair;
    int i = 0;
    for ( ; i < KNeighbors; ++i) {
        queueDistanceLabelPair.push(std::make_pair(MeasureDistanceCoordinates(vectorToPredict, coordinateMatrix[i]), labels[i]));
    }

    for ( ; i < coordinateMatrix.size(); ++i) {
        double distanceOfIthString = MeasureDistanceCoordinates(vectorToPredict, coordinateMatrix[i]);
        if (distanceOfIthString < queueDistanceLabelPair.top().first) {
            queueDistanceLabelPair.pop();
            queueDistanceLabelPair.push(std::make_pair(distanceOfIthString, labels[i]));
        }
    }

    std::unordered_map<size_t, size_t> labelsCount;
    while (!queueDistanceLabelPair.empty()) {
        ++labelsCount[queueDistanceLabelPair.top().second];
        queueDistanceLabelPair.pop();
    }

    size_t PredictedLabel = labelsCount.begin()->first;
    size_t labelMaxFrequency = labelsCount.begin()->second;
    for (const auto& x : labelsCount) {
        if (x.second > labelMaxFrequency) {
            PredictedLabel = x.first;
            labelMaxFrequency = x.second;
        }
    }

    return PredictedLabel;
}

std::vector<int> KNeighborsClassifier::Predict(const Matrix& matrixToPredict) const {
    CheckInputPredict(matrixToPredict);
    std::vector<int> output;
    output.reserve(matrixToPredict.size());
    for (const auto& x : matrixToPredict) {
        output.push_back(PredictSingleCoordinate(x));
    }
    return output;
}

std::vector<int> CreateLabels(const std::string documentName) {
    std::ifstream document(documentName);
    std::string line;
    std::vector<int> output;
    while (std::getline(document, line)) {
        output.push_back(std::stoi(line));
    }
    return output;
}

Matrix CreateMatrix(const std::string documentName) {
    std::ifstream document(documentName);
    std::string line;
    Matrix output;
    while (std::getline(document, line)) {
        Split(line, ',', &output);
    }
    return output;
}

void Split(const std::string& line, char delimiter, Matrix* matrixObject) {
    std::vector<double> coordinate;
    size_t begin = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == delimiter) {
            std::string value = line.substr(begin, i - begin);
            coordinate.push_back(std::stod(value));
            ++i;
            begin = i;
        }
    }
    std::string value = line.substr(begin);
    coordinate.push_back(stod(value));
    matrixObject->push_back(coordinate);
}

double Accuracy(std::vector<int> first, std::vector<int> second) {
    size_t correctCount = thisIsZero;
    for (int i = 0; i < first.size(); ++i) {
        if (first[i] == second[i]) {
            ++correctCount;
        }
    }
    return static_cast<double>(correctCount) / static_cast<double>(first.size());
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void TrainTestSplit(const Matrix& inputMatrix, const std::vector<int>& inputLabels,
                            Matrix* trainMatrix, std::vector<int>* trainLabels,
                            Matrix* testMatrix, std::vector<int>* testLabels, double ratioAllocateToTrain) {
    std::random_device rd;
    size_t quantityToTrain = inputLabels.size() * ratioAllocateToTrain;
    std::cout << "quantityToTrain == " << quantityToTrain << "\n";    
    std::set<size_t> indexesOfInputObjectsUsedToTrain;
    
    for (int i = 0; i < quantityToTrain; ++i) {
        size_t randomIndex = rd() % inputLabels.size();
        trainMatrix->push_back(inputMatrix[randomIndex]);
        trainLabels->push_back(inputLabels[randomIndex]);
        indexesOfInputObjectsUsedToTrain.insert(randomIndex);
    }

    for (int i = 0; i < inputLabels.size(); ++i) {
        if (indexesOfInputObjectsUsedToTrain.find(i) == indexesOfInputObjectsUsedToTrain.end()) {
            testLabels->push_back(inputLabels[i]);
            testMatrix->push_back(inputMatrix[i]);
        }
    }
}



