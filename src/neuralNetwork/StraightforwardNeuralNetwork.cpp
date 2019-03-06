#include "StraightforwardNeuralNetwork.h"
#include "layer/perceptron/activationFunction/activationFunction.h"
#include <thread>
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
                                                           float learningRate,
                                                           float momentum)
	: NeuralNetwork(structureOfNetwork,
	                activationFunctionByLayer,
	                learningRate,
	                learningRate)
{
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
	Data *data = straightforwardData.data;
	this->numberOfTrainingsBetweenTwoEvaluations =  data->sets[training].size;

	for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
	{
		this->evaluate(straightforwardData);
		//emit updateNumberOfIteration();
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
	Data *data = straightforwardData.data;
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
	/*if (this->clusteringRate > this->clusteringRateMax)
	{
		this->clusteringRateMax = this->clusteringRate;
		if (autoSave)
			this->autoSave(autoSaveFileName);
	}*/
}


void StraightforwardNeuralNetwork::trainingStop()
{
	this->wantToStopTraining = true;
	if(this->thread.joinable())
		this->thread.join();
	//this->clusteringRateMax = 0.0f;
	this->currentIndex = 0;
	this->numberOfIteration = 0;
}
