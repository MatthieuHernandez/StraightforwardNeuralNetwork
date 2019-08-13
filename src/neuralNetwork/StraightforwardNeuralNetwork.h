#pragma once
#include "neuralNetwork.h"
#include "../data/Data.h"
#include "StraightforwardOption.h"
#include "layer/perceptron/activationFunction/activationFunction.h"
#include <string>
#include <vector>
#include <thread>
#include <boost/serialization/base_object.hpp>

namespace snn
{
	class StraightforwardNeuralNetwork final : public NeuralNetwork
	{
	private :
		std::thread thread;

		bool wantToStopTraining = true;
		int currentIndex = 0;
		int numberOfIteration = 0;
		int numberOfTrainingsBetweenTwoEvaluations = 0;

		void train(Data& data);
		void (* selectEvaluationFunction(Data& data))(std::vector<float>, int);

		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:
		explicit StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork);

		StraightforwardNeuralNetwork(const std::vector<int>& structureOfNetwork,
		                             const std::vector<activationFunctionType>& activationFunctionByLayer,
									 StraightforwardOption& option);


		StraightforwardNeuralNetwork(StraightforwardNeuralNetwork& neuralNetwork);

		StraightforwardNeuralNetwork() = default;
		~StraightforwardNeuralNetwork() = default;

		StraightforwardOption option;

		void trainingStart(Data& data);
		void trainingStop();

		void evaluate(Data& straightforwardData);

		std::vector<float> computeOutput(std::vector<float> inputs);
		int computeCluster(std::vector<float> inputs);

		bool isTraining() const { return wantToStopTraining; }

		void saveAs(std::string filePath);
		static StraightforwardNeuralNetwork& loadFrom(std::string filePath);

		float getGlobalClusteringRate() const { return this->NeuralNetwork::getGlobalClusteringRate(); }
		float getWeightedClusteringRate() const { return this->NeuralNetwork::getWeightedClusteringRate(); }
		float getF1Score() const { return this->NeuralNetwork::getF1Score(); }

		int getCurrentIndex() const { return this->currentIndex; }
		int getNumberOfIteration() const { return this->numberOfIteration; }
		int getNumberOfTrainingsBetweenTwoEvaluations() const { return this->numberOfTrainingsBetweenTwoEvaluations; }

		void setNumberOfTrainingsBetweenTwoEvaluations(int value)
		{
			this->numberOfTrainingsBetweenTwoEvaluations = value;
		}

		StraightforwardNeuralNetwork& operator=(StraightforwardNeuralNetwork& neuralNetwork);
		bool operator==(const StraightforwardNeuralNetwork& neuralNetwork) const;
		bool operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const;
	};

	template <class Archive>
	void StraightforwardNeuralNetwork::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<StraightforwardNeuralNetwork, NeuralNetwork>();
		ar & boost::serialization::base_object<NeuralNetwork>(*this);
	}
}
