#pragma once
#include <vector>
#include "neuralNetwork/neuralNetwork.h"
#include "neuralNetwork/layer/perceptron/activationFunction/activationFunction.h"

namespace snn
{
	class StraightforwardNeuralNetwork final : protected NeuralNetwork
	{
	private :


	public:

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork);

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
		                             const std::vector<activationFunctionType>& activationFunctionByLayer,
		                             float learningRate = 0.05f,
		                             float momentum = 0.0f);
		~StraightforwardNeuralNetwork() = default;

		float getGlobalClusteringRate() const;
		float getWeightedClusteringRate() const;
		float getF1Score() const;
	};
}
