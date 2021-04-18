#pragma once
#include <string>
#include <vector>
#include <thread>
#include "NeuralNetwork.hpp"
#include "Wait.hpp"
#include "../data/Data.hpp"
#include "layer/LayerModel.hpp"
#include "layer/LayerFactory.hpp"
#include "optimizer/NeuralNetworkOptimizerFactory.hpp"


namespace snn
{
    class StraightforwardNeuralNetwork final : public internal::NeuralNetwork
    {
    private:
        std::thread thread;

        bool wantToStopTraining = false;
        bool isIdle = true;
        int index = 0;
        int epoch = 0;
        int numberOfTrainingsBetweenTwoEvaluations = 0;

        void resetTrainingValues();

        void trainSync(Data& data, Wait wait, int batchSize, int evaluationFrequency);
        void saveSync(std::string filePath);
        void evaluateOnce(const Data& data);

        bool continueTraining(Wait wait) const;
        void validData(const Data& data, int batchSize) const;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        StraightforwardNeuralNetwork() = default; // use restricted to Boost library only
        explicit StraightforwardNeuralNetwork(std::vector<LayerModel> architecture,
                                              NeuralNetworkOptimizerModel optimizer = {
                                                  neuralNetworkOptimizerType::stochasticGradientDescent, 0.03f, 0.0f
                                              });
        StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork);
        ~StraightforwardNeuralNetwork();

        bool autoSaveWhenBetter = false;
        std::string autoSaveFilePath = "AutoSave.snn";

        [[nodiscard]] int isValid() const;

        void startTrainingAsync(Data& data, int batchSize = 1, int evaluationFrequency = 1);
        void stopTrainingAsync();

        void waitFor(Wait wait) const;
        void train(Data& data, Wait wait, int batchSize = 1, int evaluationFrequency = 1);

        void evaluate(const Data& data);

        std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset = false);
        int computeCluster(const std::vector<float>& inputs, bool temporalReset = false);

        bool isTraining() const;

        void saveAs(std::string filePath);
        void saveFilterLayersAsBitmap(std::string filePath);
        static StraightforwardNeuralNetwork& loadFrom(std::string filePath);

        int getCurrentIndex() const { return this->index; }
        int getCurrentEpoch() const { return this->epoch; }
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
        ar & this->wantToStopTraining;
        ar & this->index;
        ar & this->isIdle;
        ar & this->epoch;
        ar & this->numberOfTrainingsBetweenTwoEvaluations;
    }
}
