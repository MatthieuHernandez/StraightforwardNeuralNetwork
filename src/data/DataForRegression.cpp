#include "DataForRegression.hpp"

using namespace std;
using namespace snn;

DataForRegression::DataForRegression(std::vector<std::vector<float>> trainingInputs,
                                     std::vector<std::vector<float>> trainingLabels,
                                     std::vector<std::vector<float>> testingInputs,
                                     std::vector<std::vector<float>> testingLabels,
                                     const float precision,
                                     temporalType type,
                                     int numberOfRecurrence)
    : Data(trainingInputs, trainingLabels, testingInputs, testingLabels, precision, type, numberOfRecurrence)
{
}

DataForRegression::DataForRegression(std::vector<std::vector<float>> inputs,
                                     std::vector<std::vector<float>> labels,
                                     const float precision,
                                     temporalType type,
                                     int numberOfRecurrence)
    : Data(inputs, labels, precision, type, numberOfRecurrence)
{
}

const vector<float>& DataForRegression::getTestingOutputs(const int index)
{
    return this->sets[testing].labels[index];
}
