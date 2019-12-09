#include "DataForRegression.hpp"
using namespace std;
using namespace snn;

DataForRegression::DataForRegression(std::vector<std::vector<float>> trainingInputs,
                                     std::vector<std::vector<float>> trainingLabels,
                                     std::vector<std::vector<float>> testingInputs,
                                     std::vector<std::vector<float>> testingLabels,
                                     const float precision)
	: Data(trainingInputs, trainingLabels, testingInputs, testingLabels, precision)
{
}

DataForRegression::DataForRegression(std::vector<std::vector<float>> inputs,
                                     std::vector<std::vector<float>> labels,
                                     const float precision)
	: Data(inputs, labels, precision)
{
}

const vector<float>& DataForRegression::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
}