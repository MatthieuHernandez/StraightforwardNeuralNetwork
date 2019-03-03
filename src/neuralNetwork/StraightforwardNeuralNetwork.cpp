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

void StraightforwardNeuralNetwork::trainingStart(StraightforwardData data)
{
	std::thread thread(&StraightforwardNeuralNetwork::train, this);
	thread.join();
}

void StraightforwardNeuralNetwork::train()
{
	this->stop = false;
	int numberOfIteration = 0;

	for (numberOfIteration = 0; !(this->stop); numberOfIteration++)
	{
		/*this->evaluate(stop, *autoSave, autoSaveFileName);
		emit updateNumberOfIteration();
		data->shuffle();

		for (outputs.currentIndex = 0; outputs.currentIndex < this->inputs.numberOfTrainbyRating && !(* stop ) ;  
		outputs . 
 			      currentIndex ++ ) 
 			 { 
 				 neuralNetwork -> train ( data -> getTrainingData ( outputs . currentIndex ) , 
 				                      data -> getTrainingOutputs ( outputs . currentIndex ) ) ; 
 			 } */
	}
}


void StraightforwardNeuralNetwork::trainingStop()
{
	throw std::exception();
}
