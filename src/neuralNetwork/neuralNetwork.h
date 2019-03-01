#pragma once
#include "layer/layer.h"
#include "layer/perceptron/activationFunction/activationFunction.h"
#include "statisticAnalysis.h"

class NeuralNetwork : public StatisticAnalysis
{
private :

	static bool isTheFirst;
	static void initialize();

	float maxOutputValue{};
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
	std::vector<float> outputs{};

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

	void train(const std::vector<float>& inputs, const std::vector<float>& desired);
	std::vector<float> output(const std::vector<float>& inputs);

	void evaluateForRegressionProblemWithPrecision(const std::vector<float>& inputs,
	                                                              const std::vector<float>& desired,
	                                                              float precision = 0.5f);
	void evaluateForRegressionProblemSeparateByValue(const std::vector<float>& inputs,
	                                                                const std::vector<float>& desired,
	                                                                float separator = 0.0f);
	void evaluateForClassificationProblem(const std::vector<float>& inputs, int classNumber);

	void addANeuron(int layerNumber);

	void saveAs(std::string filePath);
	static NeuralNetwork& loadFrom(std::string filePath);

	int isValid();
	int getLastError() const;

	Layer* getLayer(int layerNumber);

	NeuralNetwork& operator=(const NeuralNetwork& neuralNetwork);
	bool operator==(const NeuralNetwork& neuralNetwork) const;
	bool operator!=(const NeuralNetwork& neuralNetwork) const;

public:
	void setLearningRate(float learningRate);
	float getLearningRate() const;
	void setMomentum(float value);

	float getMomentum() const;
	int getNumberOfInputs() const;
	int getNumberOfHiddenLayers() const;
	int getNumberOfNeuronsInLayer(int layerNumber) const;	
	activationFunctionType getActivationFunctionInLayer(int layerNumber) const;
	int getNumberOfOutputs() const;
};

class notImplementedException : public std::exception
{
public:
	notImplementedException() : std::exception("Function not yet implemented")
	{
	}
};
