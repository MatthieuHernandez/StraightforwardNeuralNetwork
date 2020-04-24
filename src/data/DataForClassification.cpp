#include <stdexcept>
#include "DataForClassification.hpp"

using namespace std;
using namespace snn;

DataForClassification::DataForClassification(std::vector<std::vector<float>> trainingInputs,
                                             std::vector<std::vector<float>> trainingLabels,
                                             std::vector<std::vector<float>> testingInputs,
                                             std::vector<std::vector<float>> testingLabels,
                                             float separator,
                                             temporalType type,
                                             int numberOfRecurrence)
    : Data(trainingInputs, trainingLabels, testingInputs, testingLabels, separator, type, numberOfRecurrence)
{
}

DataForClassification::DataForClassification(std::vector<std::vector<float>> inputs,
                                             std::vector<std::vector<float>> labels,
                                             float separator,
                                             temporalType type,
                                             int numberOfRecurrence)
    : Data(inputs, labels, separator, type, numberOfRecurrence)
{
}

int DataForClassification::getTrainingLabel(const int index)
{
    for (int i = 0; i < this->numberOfLabel; i++)
    {
        if (this->sets[training].labels[indexes[index]][i] == 1)
            return i;
    }
    throw runtime_error("wrong label");
}

int DataForClassification::getTestingLabel(const int index)
{
    for (int i = 0; i < this->numberOfLabel; i++)
    {
        if (this->sets[testing].labels[index][i] == 1)
            return i;
    }
    throw runtime_error("wrong label");
}

const vector<float>& DataForClassification::getTestingOutputs(const int index)
{
    // Should never be called
    throw exception();
}
