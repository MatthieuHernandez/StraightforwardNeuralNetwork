#include <fstream>
#include <thread>
#include <stdexcept>
#pragma warning(push, 0)
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#pragma warning(pop)
#include "StraightforwardNeuralNetwork.hpp"
#include "StraightforwardOption.hpp"
#include "../data/DataForClassification.hpp"
#include "../data/DataForRegression.hpp"
#include "../data/DataForMultipleClassification.hpp"

using namespace std;
using namespace snn;
using namespace internal;

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(vector<int> structureOfNetwork)
	: StraightforwardNeuralNetwork(structureOfNetwork,
	                vector<activationFunctionType>(structureOfNetwork.size() - 1, sigmoid),
	                StraightforwardOption())
{
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(vector<int> structureOfNetwork,
                                                           vector<activationFunctionType> activationFunctionByLayer,
                                                           StraightforwardOption option)
	: NeuralNetwork(structureOfNetwork,
	                activationFunctionByLayer,
	                nullptr)
{
	this->option = option;
	*this->NeuralNetwork::option = *(&this->option);
	int err = this->isValid();
	if (err != 0)
	{
		string message = string("Error ") + to_string(err) + ": Wrong parameter in the creation of neural networks";
		throw runtime_error(message);
	}
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(StraightforwardNeuralNetwork& neuralNetwork)
	: NeuralNetwork(neuralNetwork)
{
	this->operator=(neuralNetwork);
}

vector<float> StraightforwardNeuralNetwork::computeOutput(const vector<float>& inputs)
{
	return this->output(inputs);
}

int StraightforwardNeuralNetwork::computeCluster(const vector<float>& inputs)
{
	const auto outputs = this->output(inputs);
	float maxOutputValue = -2;
	int maxOutputIndex = -1;
	for (int i = 0; i < outputs.size(); i++)
	{
		if (maxOutputValue < outputs[i])
		{
			maxOutputValue = outputs[i];
			maxOutputIndex = i;
		}
	}
	return maxOutputIndex;
}

void StraightforwardNeuralNetwork::trainingStart(Data& data)
{
	this->trainingStop();
	this->thread = std::thread(&StraightforwardNeuralNetwork::train, this, std::ref(data));
	this->thread.detach();
}

void StraightforwardNeuralNetwork::train(Data& data)
{
	this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
	this->wantToStopTraining = false;

	for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
	{
		this->evaluate(data);
		data.shuffle();

		for (currentIndex = 0; currentIndex < this->numberOfTrainingsBetweenTwoEvaluations && !this->wantToStopTraining;
		     currentIndex ++)
		{
			this->trainOnce(data.getTrainingData(currentIndex),
			                data.getTrainingOutputs(currentIndex));
		}
	}
}

void StraightforwardNeuralNetwork::evaluate(Data& data)
{
	const auto evaluation = selectEvaluationFunction(data);

	this->startTesting();
	for (currentIndex = 0; currentIndex < data.sets[testing].size; currentIndex++)
	{
		if (this->wantToStopTraining)
		{
			this->stopTesting();
			return;
		}

		std::invoke(evaluation, this, data);
	}
	this->stopTesting();
	if (this->option.autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
	{
			this->saveAs(option.saveFilePath);
	}
}

inline
StraightforwardNeuralNetwork::evaluationFunctionPtr StraightforwardNeuralNetwork::selectEvaluationFunction(Data& data)
{
	if(typeid(data) == typeid(DataForRegression))
	{
		return &StraightforwardNeuralNetwork::evaluateOnceForRegression;
	}
	if(typeid(data) == typeid(DataForMultipleClassification))
	{
		this->separator = data.getValue();
		return &StraightforwardNeuralNetwork::evaluateOnceForMultipleClassification;
	}
	if(typeid(data) == typeid(DataForClassification))
	{
		return &StraightforwardNeuralNetwork::evaluateOnceForClassification;
	}

	throw runtime_error("wrong Data typeid");
}

inline
void StraightforwardNeuralNetwork::evaluateOnceForRegression(Data& data)
{
	this->NeuralNetwork::evaluateOnceForRegression(
				data.getTestingData(this->currentIndex),
				data.getTestingOutputs(this->currentIndex), data.getValue());
}

inline
void StraightforwardNeuralNetwork::evaluateOnceForMultipleClassification(Data& data)
{
	this->NeuralNetwork::evaluateOnceForMultipleClassification(
				data.getTestingData(this->currentIndex),
				data.getTestingOutputs(this->currentIndex), data.getValue());
}

inline
void StraightforwardNeuralNetwork::evaluateOnceForClassification(Data& data)
{
	this->NeuralNetwork::evaluateOnceForClassification(
				data.getTestingData(this->currentIndex),
				data.getTestingLabel(this->currentIndex));
}


void StraightforwardNeuralNetwork::trainingStop()
{
	this->wantToStopTraining = true;
	if (this->thread.joinable())
		this->thread.join();
	this->currentIndex = 0;
	this->numberOfIteration = 0;
}


void StraightforwardNeuralNetwork::saveAs(std::string filePath)
{
	ofstream file(filePath);
	boost::archive::text_oarchive binaryFile(file);
	binaryFile << this;
	std::ofstream ofs(filePath);
	boost::archive::text_oarchive oa(ofs);
}

StraightforwardNeuralNetwork& StraightforwardNeuralNetwork::loadFrom(std::string filePath)
{
	StraightforwardNeuralNetwork* neuralNetwork{};
	ifstream file(filePath, ios::binary);
	boost::archive::text_iarchive binaryFile(file);
	binaryFile >> neuralNetwork;
	return *neuralNetwork;
}

StraightforwardNeuralNetwork& StraightforwardNeuralNetwork::operator=(StraightforwardNeuralNetwork& neuralNetwork)
{
	this->trainingStop();
	neuralNetwork.trainingStop();
	this->option = neuralNetwork.option;
	this->currentIndex = neuralNetwork.currentIndex;
	this->numberOfIteration = neuralNetwork.numberOfIteration;
	this->numberOfTrainingsBetweenTwoEvaluations = neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
	this->NeuralNetwork::operator=(neuralNetwork);
	return *this;
}

bool StraightforwardNeuralNetwork::operator==(const StraightforwardNeuralNetwork& neuralNetwork) const
{
	return this->NeuralNetwork::operator==(neuralNetwork) && this->option == neuralNetwork.option;
}

bool StraightforwardNeuralNetwork::operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const
{
	return !this->operator==(neuralNetwork);
}
