#include "neuralNetwork.h"

using namespace std;

vector<float> NeuralNetwork::output(const vector<float>& inputs)
{
	this->outputs = layers[0]->output(inputs);

	for (int l = 1; l < numberOfLayers; ++l)
	{
		outputs = layers[l]->output(outputs);
	}
	return outputs;
}

void NeuralNetwork::evaluateForRegressionProblemWithPrecision(
	const vector<float>& inputs, const vector<float>& desired, float precision)
{
	this->outputs = this->output(inputs);
	this->insertTestWithPrecision(this->outputs, desired, precision);
}

void NeuralNetwork::evaluateForRegressionProblemSeparateByValue(
	const vector<float>& inputs, const vector<float>& desired, float separator)
{
	this->outputs = this->output(inputs);
	this->insertTestSeparateByValue(this->outputs, desired, separator);
}

void NeuralNetwork::evaluateForClassificationProblem(const vector<float>& inputs, int classNumber)
{
	maxOutputValue = -1;
	this->outputs = this->output(inputs);
	this->insertTestWithClassNumber(this->outputs, classNumber);
}

void NeuralNetwork::train(const vector<float>& inputs, const vector<float>& desired)
{
	backpropagationAlgorithm(inputs, desired);
}

void NeuralNetwork::backpropagationAlgorithm(const vector<float>& inputs, const vector<float>& desired)
{
	this->outputs = this->output(inputs);
	auto errors = calculateError(this->outputs, desired);

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
