#pragma once
#include <string>
#include <vector>
#include <thread>
#include "NeuralNetwork.hpp"
#include "../data/Data.hpp"
#include "layer/LayerModel.hpp"
#include "layer/LayerFactory.hpp"
#include "layer/perceptron/activation_function/ActivationFunction.hpp"

namespace snn
{
	class StraightforwardNeuralNetwork final : public internal::NeuralNetwork
	{
	private :
		std::thread thread;

		bool wantToStopTraining = false;
		bool isIdle = true;
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
		StraightforwardNeuralNetwork() = default; // use restricted to Boost library only
		StraightforwardNeuralNetwork(int numberOfInputs, std::vector<LayerModel> models);
		StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork);
		~StraightforwardNeuralNetwork() = default;

		bool autoSaveWhenBetter = false;
		std::string autoSaveFilePath = "AutoSave.snn";

		int isValid() const;

		void trainingStart(Data& data);
		void trainingStop();

		void evaluate(Data& straightforwardData);

		std::vector<float> computeOutput(const std::vector<float>& inputs);
		int computeCluster(const std::vector<float>& inputs);

		bool isTraining() const { return wantToStopTraining; }

		void saveAs(std::string filePath);
		static StraightforwardNeuralNetwork loadFrom(std::string filePath);

		int getCurrentIndex() const { return this->currentIndex; }
		int getNumberOfIteration() const { return this->numberOfIteration; }
		int getNumberOfTrainingsBetweenTwoEvaluations() const { return this->numberOfTrainingsBetweenTwoEvaluations; }

		void setNumberOfTrainingsBetweenTwoEvaluations(int value)
		{
			this->numberOfTrainingsBetweenTwoEvaluations = value;
		}

		bool operator==(const StraightforwardNeuralNetwork& neuralNetwork) const;
		bool operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const;
	};

	template <class Archive>
	void StraightforwardNeuralNetwork::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<StraightforwardNeuralNetwork, NeuralNetwork>();
		ar & boost::serialization::base_object<NeuralNetwork>(*this);
		ar & this->autoSaveFilePath;
		ar & this->autoSaveWhenBetter;
	}
}
