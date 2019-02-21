#pragma once
#include "perceptron.h"
#include <boost/serialization/access.hpp>


enum LayerType
{
	allToAll = 0
};

class Layer
{
private :

	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, unsigned version);


protected:

	int numberOfInputs = 0;
	int numberOfNeurons = 0;
	std::vector<float> errors;
	std::vector<float> outputs;
	std::vector<Perceptron> neurons;
	float learningRate = 0;
	float momentum = 0;


public:


	virtual ~Layer() = default;

	virtual std::vector<float>& output(const std::vector<float>& inputs) = 0;
	virtual std::vector<float>& backOutput(std::vector<float>& inputsError) = 0;
	virtual void train(std::vector<float>& inputsError) = 0;

	Perceptron* getNeuron(int neuronNumber);
	virtual LayerType getType() const = 0;

	virtual Layer& equal(const Layer& layer) = 0;
	virtual bool operator==(const Layer& layer) const;
	virtual bool operator!=(const Layer& layer) const;
};

template <class Archive>
void Layer::serialize(Archive& ar, unsigned version)
{
	ar & this->numberOfInputs;
	ar & this->numberOfNeurons;
	ar & this->errors;
	ar & this->outputs;
	ar & this->learningRate;
	ar & this->momentum;
	ar & this->neurons;
}
