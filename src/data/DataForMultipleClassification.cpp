#include "DataForMultipleClassification.h"
#include "DataForRegression.h"
using namespace std;
using namespace snn;

DataForMultipleClassification::DataForMultipleClassification(std::vector<std::vector<float>>& trainingInputs,
                                                             std::vector<std::vector<float>>& trainingLabels,
                                                             std::vector<std::vector<float>>& testingInputs,
                                                             std::vector<std::vector<float>>& testingLabels,
                                                             float separator)
	: Data(trainingInputs, trainingLabels, testingInputs, testingLabels, separator)
{
}

DataForMultipleClassification::DataForMultipleClassification(std::vector<std::vector<float>>& inputs,
                                                             std::vector<std::vector<float>>& labels,
                                                             float separator)
	: Data(inputs, labels, separator)
{
}

vector<float>& DataForMultipleClassification::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
}