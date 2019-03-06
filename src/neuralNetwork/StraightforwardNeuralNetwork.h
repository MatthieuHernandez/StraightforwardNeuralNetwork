#pragma once
#include <vector>
#include "neuralNetwork.h"
#include "../data/StraightforwardData.h" // Do not remove
#include "layer/perceptron/activationFunction/activationFunction.h"
#include <thread>

namespace snn
{
	class StraightforwardNeuralNetwork final : public NeuralNetwork
	{
	private :

		std::thread thread;

		bool wantToStopTraining = true;
		int currentIndex = 0;
		int numberOfIteration = 0;
		float clusteringRateMax = -1.0f;
		float weightedClusteringRateMax = -1.0f;
		float f1ScoreMax = -1.0f;
		int numberOfTrainingsBetweenTwoEvaluations = 0;

		void train(StraightforwardData& data);
		//void trainOnce();
		void evaluate(StraightforwardData& straightforwardData);


	public:

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork);

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
		                             const std::vector<activationFunctionType>& activationFunctionByLayer,
		                             float learningRate = 0.05f,
		                             float momentum = 0.0f);
		~StraightforwardNeuralNetwork() = default;

		void trainingStart(StraightforwardData& data);
		void trainingStop();

		std::vector<float> computeOutput(std::vector<float> inputs);
		int computeCluster(std::vector<float> inputs);


		float getGlobalClusteringRate() const { return this->NeuralNetwork::getGlobalClusteringRate(); }
		float getWeightedClusteringRate() const { return this->NeuralNetwork::getWeightedClusteringRate(); }
		float getF1Score() const { return this->NeuralNetwork::getF1Score(); }
	};
}
