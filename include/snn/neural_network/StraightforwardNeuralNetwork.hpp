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
        void evaluate(const Data& data, Wait* wait);
        void evaluateOnce(const Data& data);

        bool continueTraining(Wait wait) const;
        void validData(const Data& data, int batchSize) const;

        template <logLevel T>
        void logAccuracy(Wait& wait, const bool hasSaved) const;
        template <logLevel T>
        void logInProgress(Wait& wait, const Data& data, set set) const;

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

        void evaluate(const Data& data) { return this->evaluate(data, nullptr); }

        std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset = false);
        int computeCluster(const std::vector<float>& inputs, bool temporalReset = false);

        bool isTraining() const;

        void saveAs(std::string filePath);
        void saveFeatureMapsAsBitmap(std::string filePath);
        void saveData2DAsBitmap(std::string filePath, const Data& data, int dataIndex);
        void saveFilterLayersAsBitmap(std::string filePath, const Data& data, int dataIndex);
        static StraightforwardNeuralNetwork& loadFrom(std::string filePath);

        [[nodiscard]] std::string summary() const;

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


    template <logLevel T>
    void StraightforwardNeuralNetwork::logAccuracy(Wait& wait, const bool hasSaved) const
    {
        if constexpr (T > none && T <= verbose)
        {
            tools::log<T, false>("\rEpoch: ", tools::toConstSizeString(this->epoch, 2),
                                " - Accuracy: ", tools::toConstSizeString<2>(this->getGlobalClusteringRate(), 4),
                                " - MAE: ", tools::toConstSizeString<4>(this->getMeanAbsoluteError(), 7),
                                " - Time: ", tools::toConstSizeString<0>(wait.getDurationAndReset(), 2), "s");
            if (hasSaved)
                tools::log<T, false>(" - Saved");
            tools::log<T>();
        }
    }

    template <logLevel T>
    void StraightforwardNeuralNetwork::logInProgress(Wait& wait, const Data& data, set set) const
    {
        if constexpr (T > none && T <= verbose)
        {
            if (wait.tick() >= 100)
            {
                const std::string name = set == training ? "Training " : "Evaluation";
                const int progress = static_cast<int>(this->index / static_cast<float>(data.sets[set].size) * 100);
                tools::log<T, false>("\rEpoch: ", tools::toConstSizeString(this->epoch, 2),
                    " - ", name, "in progress...   ", tools::toConstSizeString(progress, 2), "%",
                    " - Time: ", tools::toConstSizeString<0>(wait.getDuration(), 2), "s");
            }
        }
    }

    template <class Archive>
    void StraightforwardNeuralNetwork::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
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
