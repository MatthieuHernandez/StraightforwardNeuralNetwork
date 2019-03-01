#pragma once
#include <vector>
#include "neuralNetwork.h"
#include "../data/StraightforwardData.h" // Do not removegb
#include "layer/perceptron/activationFunction/activationFunction.h"

namespace snn
{
	class StraightforwardNeuralNetwork final : public NeuralNetwork
	{
	private :


	public:

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork);

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
		                             const std::vector<activationFunctionType>& activationFunctionByLayer,
		                             float learningRate = 0.05f,
		                             float momentum = 0.0f);
		~StraightforwardNeuralNetwork() = default;

		void trainingStart(StraightforwardData data);
		void trainingStop();

		std::vector<float> computeOutput(std::vector<float> inputs);
		int computeCluster(std::vector<float> inputs);


		float getGlobalClusteringRate() const;
		float getWeightedClusteringRate() const;
		float getF1Score() const;
	};
}
