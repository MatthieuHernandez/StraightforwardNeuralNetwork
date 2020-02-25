#pragma once
#include <string>
#include <vector>
#include <thread>
#include "NeuralNetwork.hpp"
#include "Wait.hpp"
#include "../data/Data.hpp"
#include "../data/DataForClassification.hpp"
#include "../data/DataForMultipleClassification.hpp"
#include "../data/DataForRegression.hpp"
#include "layer/LayerModel.hpp"
#include "layer/LayerFactory.hpp"

namespace snn
{
    class StraightforwardNeuralNetwork final : public internal::NeuralNetwork
    {
    private:
        std::thread thread;

        bool wantToStopTraining = false;
        bool isIdle = true;
        int currentIndex = 0;
        int numberOfIteration = 0;
        int numberOfTrainingsBetweenTwoEvaluations = 0;

        //TODO: Use C++20 concepts to only allow class derivative from Data
        template<class TData>
        void train(TData& data);

        void evaluateOnce(DataForRegression& data);
        void evaluateOnce(DataForMultipleClassification& data);
        void evaluateOnce(DataForClassification& data);

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version);

    public:
        StraightforwardNeuralNetwork() = default; // use restricted to Boost library only
        StraightforwardNeuralNetwork(int numberOfInputs, std::vector<LayerModel> models);
        StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork);
        ~StraightforwardNeuralNetwork();

        bool autoSaveWhenBetter = false;
        std::string autoSaveFilePath = "AutoSave.snn";

        [[nodiscard]] int isValid() const;
        [[nodiscard]] bool validData(const Data& data) const;

        template<class TData>
        void startTraining(TData& data);
        void stopTraining();

        void waitFor(Wait wait) const;

        template <typename TData>
        void evaluate(TData& data);

        std::vector<float> computeOutput(const std::vector<float>& inputs);
        int computeCluster(const std::vector<float>& inputs);

        bool isTraining() const;

        void saveAs(std::string filePath);
        static StraightforwardNeuralNetwork& loadFrom(std::string filePath);

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
