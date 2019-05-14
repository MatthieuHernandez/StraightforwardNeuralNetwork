#include "StraightforwardNeuralNetwork.h"
#include "StraightforwardOption.h"
#include "layer/perceptron/activationFunction/activationFunction.h"
#include <thread>
#include <fstream>
#pragma warning(push, 0)
#include <boost/serialization/base_object.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#pragma warning(pop)
using namespace std;
using namespace snn;

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork)
	: NeuralNetwork(structureOfNetwork,
	                std::vector<activationFunctionType>(structureOfNetwork.size() - 1, sigmoid),
	                0.05f,
	                0.0f)
{
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
                                                           const std::vector<activationFunctionType>&
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

std::vector<float> StraightforwardNeuralNetwork::computeOutput(std::vector<float> inputs)
{
	return this->output(inputs);
}

int StraightforwardNeuralNetwork::computeCluster(std::vector<float> inputs)
{
	throw std::exception();
}

void StraightforwardNeuralNetwork::trainingStart(StraightforwardData& data)
{
	this->trainingStop();
	this->thread = std::thread(&StraightforwardNeuralNetwork::train, this, std::ref(data));
	this->thread.detach();
}

void StraightforwardNeuralNetwork::train(StraightforwardData& straightforwardData)
{
	this->wantToStopTraining = false;
	Data* data = straightforwardData.data;
	this->numberOfTrainingsBetweenTwoEvaluations = data->sets[training].size;

	for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
	{
		this->evaluate(straightforwardData);
		data->shuffle();

		for (currentIndex = 0; currentIndex < this->numberOfTrainingsBetweenTwoEvaluations && !this->wantToStopTraining;
		     currentIndex ++)
		{
			this->trainOnce(data->getTrainingData(currentIndex),
			                data->getTrainingOutputs(currentIndex));
		}
	}
}

void StraightforwardNeuralNetwork::evaluate(StraightforwardData& straightforwardData)
{
	Data* data = straightforwardData.data;
	this->startTesting();
	for (currentIndex = 0; currentIndex < data->sets[testing].size; currentIndex++)
	{
		if (this->wantToStopTraining)
			return;
		if (data->problem == classification)
		{
			this->evaluateForClassificationProblem(
				data->getTestingData(this->currentIndex),
				data->getTestingLabel(this->currentIndex));
		}
		else
		{
			this->evaluateForRegressionProblemSeparateByValue(
				data->getTestingData(this->currentIndex),
				data->getTestingOutputs(this->currentIndex), 0.5f);
		}
	}
	this->stopTesting();
	if (this->option.autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
	{
			this->saveAs(option.saveFilePath);
	}
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
