#pragma once
#include "layer/layer.h"
#include "layer/alltoall.h"
#include "layer/perceptron/activationFunction/activationFunction.h"
#include "statisticAnalysis.h"

class 
NeuralNetwork : public StatisticAnalysis
{
private :
	static bool isTheFirst;
	static void initialize();

	int maxOutputIndex{};
	int lastError{};
	float learningRate{};

	float error{};
	float momentum{};

	int numberOfHiddenLayers{};
	int numberOfLayers{};
	int numberOfInput{};
	int numberOfOutputs{};

	std::vector<int> structureOfNetwork{};
	std::vector<activationFunctionType> activationFunctionByLayer{};

	std::vector<Layer*> layers{};

	std::vector<float> errors{};

	void backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired);
	std::vector<float> calculateError(const std::vector<float>& outputs, const std::vector<float>& desired);

	void resetAllNeurons();

	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, unsigned version);


protected :
	NeuralNetwork(const std::vector<int>& structureOfNetwork,
	              const std::vector<activationFunctionType>& activationFunctionByLayer,
	              float learningRate = 0.05f,
	              float momentum = 0.0f);

	NeuralNetwork(const NeuralNetwork& neuralNetwork);

	NeuralNetwork() = default;
	~NeuralNetwork() = default;

	[[nodiscard]] std::vector<float> output(const std::vector<float>& inputs);

	void evaluateOnceForRegression(const std::vector<float>& inputs,
	                                                              const std::vector<float>& desired,
	                                                              float precision);
	void evaluateOnceForMultipleClassification(const std::vector<float>& inputs,
	                                                                const std::vector<float>& desired,
	                                                                float separator);
	void evaluateOnceForClassification(const std::vector<float>& inputs, int classNumber);

	void addANeuron(int layerNumber);

	int isValid();
	int getLastError() const;

	NeuralNetwork& operator=(const NeuralNetwork& neuralNetwork);
	bool operator==(const NeuralNetwork& neuralNetwork) const;
	bool operator!=(const NeuralNetwork& neuralNetwork) const;

public:
	void trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired);

	void setLearningRate(float learningRate);
	float getLearningRate() const;
	void setMomentum(float value);

	float getMomentum() const;
	int getNumberOfInputs() const;
	int getNumberOfHiddenLayers() const;
	int getNumberOfNeuronsInLayer(int layerNumber) const;	
	Layer* getLayer(int layerNumber);
	activationFunctionType getActivationFunctionInLayer(int layerNumber) const;
	int getNumberOfOutputs() const;
};

template <class Archive>
void NeuralNetwork::serialize(Archive& ar, const unsigned int version)
{
	boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
	ar & boost::serialization::base_object<StatisticAnalysis>(*this);
	ar & this->maxOutputIndex;
	ar & this->lastError;
	ar & this->learningRate;
	ar & this->error;
	ar & this->momentum;
	ar & this->numberOfHiddenLayers;
	ar & this->numberOfLayers;
	ar & this->numberOfInput;
	ar & this->numberOfOutputs;
	ar & this->structureOfNetwork;
	ar & this->activationFunctionByLayer;
	ar & this->errors;
	ar & this->numberOfInput;

	ar.template register_type<AllToAll>();
	ar & layers;
}

class notImplementedException : public std::exception
{
public:
	notImplementedException() : std::exception("Function not yet implemented")
	{
	}
};
