#include "DataForRegression.h"
using namespace std;
using namespace snn;

DataForRegression::DataForRegression(std::vector<std::vector<float>>& trainingInputs,
                                     std::vector<std::vector<float>>& trainingLabels,
                                     std::vector<std::vector<float>>& testingInputs,
                                     std::vector<std::vector<float>>& testingLabels)
	: Data(regression, trainingInputs, trainingLabels, testingInputs, testingLabels)
{
}

vector<float>& DataForRegression::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
}
