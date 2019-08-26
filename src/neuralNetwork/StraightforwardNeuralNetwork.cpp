#include <thread>
#include <fstream>
#pragma warning(push, 0)
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#pragma warning(pop)
#include "StraightforwardNeuralNetwork.h"
#include "StraightforwardOption.h"
#include "../data/DataForClassification.h"
#include "../data/DataForRegression.h"
#include "../data/DataForMultipleClassification.h"

using namespace std;
using namespace snn;

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const vector<int>& structureOfNetwork)
	: NeuralNetwork(structureOfNetwork,
	                vector<activationFunctionType>(structureOfNetwork.size() - 1, sigmoid),
	                0.05f,
	                0.0f)
{
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const vector<int>& structureOfNetwork,
                                                           const vector<activationFunctionType>&
                                                           activationFunctionByLayer,
														   StraightforwardOption& option)
	: NeuralNetwork(structureOfNetwork,
	                activationFunctionByLayer,
	                option.learningRate,
	                option.momentum)
{
	this->option = option;
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
	auto output = this->output(inputs);

	float separator = 0.5f;
	float maxOutputValue = -1;
	int maxOutputIndex = -1;
	for (int i = 0; i < clusters.size(); i++)
	{
		if (maxOutputValue < output[i])
		{
			maxOutputValue = output[i];
			maxOutputIndex = i;
		}
		if (i == classNumber && output[i] > separator)
		{
			clusters[i].truePositive ++;
		}
		else if (i == classNumber && output[i] <= separator)
		{
			clusters[i].falseNegative ++;
		}
		else if (output[i] > separator)
		{
			clusters[i].falsePositive ++;
		}
		else if (output[i] <= separator)
		{
			clusters[i].trueNegative ++;
		}
	}
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
		return &StraightforwardNeuralNetwork::evaluateOnceForMultipleClassification;
	}
	if(typeid(data) == typeid(DataForClassification))
	{
		return &StraightforwardNeuralNetwork::evaluateOnceForClassification;
	}

	throw exception("wrong Data typeid");
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
	std::ofstream ofs("filename");
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

	this->NeuralNetwork::operator=(neuralNetwork);
	this->currentIndex = neuralNetwork.currentIndex;
	this->numberOfIteration = neuralNetwork.numberOfIteration;
	this->numberOfTrainingsBetweenTwoEvaluations = neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
	return *this;
}

bool StraightforwardNeuralNetwork::operator==(const StraightforwardNeuralNetwork& neuralNetwork) const
{
	return this->NeuralNetwork::operator==(neuralNetwork);
}

bool StraightforwardNeuralNetwork::operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const
{
	return this->NeuralNetwork::operator!=(neuralNetwork);
}
