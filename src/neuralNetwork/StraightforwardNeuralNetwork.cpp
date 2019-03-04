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
	throw std::exception();
}

int StraightforwardNeuralNetwork::computeCluster(std::vector<float> inputs)
{
	throw std::exception();
}

void StraightforwardNeuralNetwork::trainingStart(StraightforwardData& data)
{
	std::thread thread(&StraightforwardNeuralNetwork::train, this, data);
	thread.join();
}

void StraightforwardNeuralNetwork::train(StraightforwardData& straightforwardData)
{
	this->isTraining = false;
	Data data = *straightforwardData.data;

	for (this->numberOfIteration = 0; this->isTraining; this->numberOfIteration++)
	{
		this->evaluate(straightforwardData);
		//emit updateNumberOfIteration();
		straightforwardData.data->shuffle();

		for (currentIndex = 0; currentIndex < this->numberOfTrainingsBetweenTwoEvaluations && !this->isTraining;
		     currentIndex ++)
		{
			this->trainOnce(data.getTrainingData(currentIndex),
			                data.getTrainingOutputs(currentIndex));
		}
	}
}

/*void StraightforwardNeuralNetwork::trainOnce()
{
	this->train();
}*/

void StraightforwardNeuralNetwork::evaluate(StraightforwardData& straightforwardData)
{
	Data data = *straightforwardData.data;

	this->startTesting();
	for (currentIndex = 0; currentIndex < data.sets[testing].size; currentIndex++)
	{
		if (!this->isTraining)
			return;
		if (data.problem == classification)
		{
			this->evaluateForClassificationProblem(
				data.getTestingData(this->currentIndex),
				data.getTestingLabel(this->currentIndex));
		}
		else
		{
			this->evaluateForRegressionProblemSeparateByValue(
				data.getTestingData(this->currentIndex),
				data.getTestingOutputs(this->currentIndex), 0.0f);
		}
	}
	this->clusteringRate = this->getGlobalClusteringRate();
	this->weightedClusteringRate = this->getWeightedClusteringRate();
	this->f1Score = this->getF1Score();
	if (this->clusteringRate > this->clusteringRateMax)
	{
		this->clusteringRateMax = this->clusteringRate;
		/*if (autoSave)
			this->autoSave(autoSaveFileName);*/
	}
}


void StraightforwardNeuralNetwork::trainingStop()
{
	throw std::exception();
}
