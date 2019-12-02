#pragma once
#include <memory>
#include <vector>
#include "NeuronOption.hpp"
#include "activation_function/ActivationFunction.hpp"

namespace snn::internal
{
	class Perceptron
	{
	private :

		int numberOfInputs{};

		NeuronOption* option;

		std::vector<float> weights;
		float bias{};

		std::vector<float> previousDeltaWeights;
		std::vector<float> lastInputs;
		std::vector<float> errors;

		float lastOutput{};

		activationFunctionType aFunctionType{};
		ActivationFunction* activationFunction = nullptr;

		float randomInitializeWeight() const;

		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);;


	public :

		Perceptron() = default;
		~Perceptron();
		Perceptron(int numberOfInputs, activationFunctionType activationFunction, NeuronOption* option);
		Perceptron(const Perceptron& perceptron);

		std::vector<float>& backOutput(float error);
		float output(const std::vector<float>& inputs);
		void train(const std::vector<float>& inputs, float error);

		void addAWeight();

		int isValid() const;

		ActivationFunction* getActivationFunction() const;

		std::vector<float> getWeights() const;
		void setWeights(const std::vector<float>& weights);

		float getWeight(int w) const;
		void setWeight(int w, float weight);

		float getBias() const;
		void setBias(float bias);

		int getNumberOfInputs() const;

		Perceptron& operator=(const Perceptron& perceptron);
		bool operator==(const Perceptron& perceptron) const;
		bool operator!=(const Perceptron& perceptron) const;
	};

	template <class Archive>
	void Perceptron::serialize(Archive& ar, const unsigned int version)
	{
		ar & this->option;
		ar & this->weights;
		ar & this->previousDeltaWeights;
		ar & this->lastInputs;
		ar & this->errors;
		ar & this->lastOutput;
		ar & this->numberOfInputs;
		ar & this->bias;
		ar & this->aFunctionType;
		this->activationFunction = ActivationFunction::create(aFunctionType);
	}
}
