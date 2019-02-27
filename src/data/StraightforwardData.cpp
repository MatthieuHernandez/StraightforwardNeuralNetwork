#include "StraightforwardData.h"
#include "data/DataForClassification.h"
#include "data/DataForRegression.h"
using namespace std;
using namespace snn;


StraightforwardData::StraightforwardData(problemType type,
                                         std::vector<std::vector<float>>& Inputs,
                                         std::vector<std::vector<float>>& Labels)
	: StraightforwardData(type, Inputs, Labels, Inputs, Labels)
{
}

StraightforwardData::StraightforwardData(problemType type,
                                         std::vector<std::vector<float>>& trainingInputs,
                                         std::vector<std::vector<float>>& trainingLabels,
                                         std::vector<std::vector<float>>& testingInputs,
                                         std::vector<std::vector<float>>& testingLabels)
{
	if (type == classification)
		this->data = new DataForClassification(trainingInputs,
		                                       trainingLabels,
		                                       testingInputs,
		                                       testingLabels);

	else if (type == regression)
		this->data = new DataForRegression(trainingInputs,
		                                   trainingLabels,
		                                   testingInputs,
		                                   testingLabels);

	else
		throw exception("wrong problem type");
}
