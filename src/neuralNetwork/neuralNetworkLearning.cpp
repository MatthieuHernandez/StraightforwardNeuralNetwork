#include "neuralNetwork.h"

using namespace std;
using namespace snn;
using namespace internal;

vector<float> NeuralNetwork::output(const vector<float>& inputs)
{
	auto outputs = layers[0]->output(inputs);

	for (int l = 1; l < numberOfLayers; ++l)
	{
		outputs = layers[l]->output(outputs);
	}
	return outputs;
}

void NeuralNetwork::evaluateOnceForRegression(
	const vector<float>& inputs, const vector<float>& desired, const float precision)
{
	const auto outputs = this->output(inputs);
	this->StatisticAnalysis::evaluateOnceForRegression(outputs, desired, precision);
}

void NeuralNetwork::evaluateOnceForMultipleClassification(
	const vector<float>& inputs, const vector<float>& desired, const float separator)
{
	const auto outputs = this->output(inputs);
	this->StatisticAnalysis::evaluateOnceForMultipleClassification(outputs, desired, separator);
}

void NeuralNetwork::evaluateOnceForClassification(const vector<float>& inputs, const int classNumber)
{
	const auto outputs = this->output(inputs);
	this->StatisticAnalysis::evaluateOnceForClassification(outputs, classNumber);
}

void NeuralNetwork::trainOnce(const vector<float>& inputs, const vector<float>& desired)
{
	this->backpropagationAlgorithm(inputs, desired);
}

void NeuralNetwork::backpropagationAlgorithm(const vector<float>& inputs, const vector<float>& desired)
{
	const auto outputs = this->output(inputs);
	this->errors = calculateError(outputs, desired);

	for (int l = numberOfLayers - 1; l > 0; --l)
	{
		errors = layers[l]->backOutput(errors);
	}
	layers[0]->train(errors);
}

inline vector<float> NeuralNetwork::calculateError(const vector<float>& outputs, const vector<float>& desired)
{
	for (int n = 0; n < numberOfOutputs; ++n)
	{
		if (desired[n] != -1.0f)
		{
			float e = desired[n] - outputs[n];
			this->errors[n] = e * abs(e);
		}
		else
			this->errors[n] = 0;
	}
	return this->errors;
}
