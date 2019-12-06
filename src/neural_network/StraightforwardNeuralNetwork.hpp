#pragma once
#include <string>
#include <vector>
#include <thread>
#include "StraightforwardOption.hpp"
#include "neuralNetwork.hpp"
#include "../data/Data.hpp"
#include "layer/perceptron/activation_function/activationFunction.hpp"

namespace snn
{
	class StraightforwardNeuralNetwork final : public internal::NeuralNetwork
	{
	private :
		std::thread thread;

		bool wantToStopTraining = false;
		int currentIndex = 0;
		int numberOfIteration = 0;
		int numberOfTrainingsBetweenTwoEvaluations = 0;

		void train(Data& data);

		typedef void (StraightforwardNeuralNetwork::* evaluationFunctionPtr)(Data& data);

		evaluationFunctionPtr selectEvaluationFunction(Data& data);
		void evaluateOnceForRegression(Data& data);
		void evaluateOnceForMultipleClassification(Data& data);
		void evaluateOnceForClassification(Data& data);

		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:
		explicit StraightforwardNeuralNetwork(std::vector<int> structureOfNetwork);

		StraightforwardNeuralNetwork(std::vector<int> structureOfNetwork,
		                             std::vector<activationFunctionType> activationFunctionByLayer,
		                             StraightforwardOption option);


		StraightforwardNeuralNetwork(StraightforwardNeuralNetwork& neuralNetwork);

		StraightforwardNeuralNetwork() = default;
		~StraightforwardNeuralNetwork() = default;

		StraightforwardOption option;

		void trainingStart(Data& data);
		void trainingStop();

		void evaluate(Data& straightforwardData);

		std::vector<float> computeOutput(const std::vector<float>& inputs);
		int computeCluster(const std::vector<float>& inputs);

		bool isTraining() const { return wantToStopTraining; }

		void saveAs(std::string filePath);
		static StraightforwardNeuralNetwork& loadFrom(std::string filePath);

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
