#include "DataForMultipleClassification.hpp"
#include "DataForRegression.hpp"

using namespace std;
using namespace snn;

DataForMultipleClassification::DataForMultipleClassification(std::vector<std::vector<float>> trainingInputs,
                                                             std::vector<std::vector<float>> trainingLabels,
                                                             std::vector<std::vector<float>> testingInputs,
                                                             std::vector<std::vector<float>> testingLabels,
                                                             float separator,
                                                             temporalType type,
                                                             int numberOfRecurrence)
    : Data(trainingInputs, trainingLabels, testingInputs, testingLabels, separator, type, numberOfRecurrence)
{
}

DataForMultipleClassification::DataForMultipleClassification(std::vector<std::vector<float>> inputs,
                                                             std::vector<std::vector<float>> labels,
                                                             float separator,
                                                             temporalType type,
                                                             int numberOfRecurrence)
    : Data(inputs, labels, separator, type, numberOfRecurrence)
{
}

const vector<float>& DataForMultipleClassification::getTestingOutputs(const int index)
{
    return this->sets[testing].labels[index];
}
