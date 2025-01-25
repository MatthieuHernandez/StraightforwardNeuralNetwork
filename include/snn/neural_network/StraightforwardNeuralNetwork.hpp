#pragma once
#include <string>
#include <thread>
#include <vector>

#include "../data/Data.hpp"
#include "../tools/Error.hpp"
#include "NeuralNetwork.hpp"
#include "Wait.hpp"
#include "layer/LayerFactory.hpp"
#include "layer/LayerModel.hpp"
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
        void evaluate(const Data& data, Wait& wait);
        void evaluateOnce(const Data& data);

        auto continueTraining(Wait wait) const -> bool;
        void validData(const Data& data, int batchSize) const;

        template <logLevel T>
        void logAccuracy(Wait& wait, const bool hasSaved) const;
        template <logLevel T>
        void logInProgress(Wait& wait, const Data& data, set set) const;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    public:
        StraightforwardNeuralNetwork() = default;  // use restricted to Boost library only
        explicit StraightforwardNeuralNetwork(std::vector<LayerModel> architecture,
                                              NeuralNetworkOptimizerModel optimizer = {
                                                  neuralNetworkOptimizerType::stochasticGradientDescent, 0.03F, 0.0F});
        StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork);
        ~StraightforwardNeuralNetwork() final;

        bool autoSaveWhenBetter = false;
        std::string autoSaveFilePath = "AutoSave.snn";

        [[nodiscard]] auto isValid() const -> ErrorType;

        void startTrainingAsync(Data& data, int batchSize = 1, int evaluationFrequency = 1);
        void stopTrainingAsync();

        void waitFor(Wait wait) const;
        void train(Data& data, Wait wait, int batchSize = 1, int evaluationFrequency = 1);

        void evaluate(const Data& data);

        auto computeOutput(const std::vector<float>& inputs, bool temporalReset = false) -> std::vector<float>;
        auto computeCluster(const std::vector<float>& inputs, bool temporalReset = false) -> int;

        auto isTraining() const -> bool;

        void saveAs(std::string filePath);
        void saveFeatureMapsAsBitmap(std::string filePath);
        void saveData2DAsBitmap(std::string filePath, const Data& data, int dataIndex);
        void saveFilterLayersAsBitmap(std::string filePath, const Data& data, int dataIndex);
        static auto loadFrom(std::string filePath) -> StraightforwardNeuralNetwork&;

        [[nodiscard]] auto summary() const -> std::string;

        auto getCurrentIndex() const -> int { return this->index; }
        auto getCurrentEpoch() const -> int { return this->epoch; }
        auto getNumberOfTrainingsBetweenTwoEvaluations() const -> int
        {
            return this->numberOfTrainingsBetweenTwoEvaluations;
        }

        void setNumberOfTrainingsBetweenTwoEvaluations(int value)
        {
            this->numberOfTrainingsBetweenTwoEvaluations = value;
        }

        auto operator==(const StraightforwardNeuralNetwork& neuralNetwork) const -> bool;
        auto operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const -> bool;
};

template <logLevel T>
void StraightforwardNeuralNetwork::logAccuracy(Wait& wait, const bool hasSaved) const
{
    if constexpr (T > none && T <= verbose)
    {
        tools::log<T, false>("\rEpoch: ", tools::toConstSizeString(this->epoch, 2),
                             " - Accuracy: ", tools::toConstSizeString<2>(this->getGlobalClusteringRate(), 4),
                             " - MAE: ", tools::toConstSizeString<4>(this->getMeanAbsoluteError(), 9),
                             " - Time: ", tools::toConstSizeString<0>(wait.getDurationAndReset(), 3), "s");
        if (hasSaved) tools::log<T, false>(" - Saved");
        tools::log<T>();
    }
}

template <logLevel T>
void StraightforwardNeuralNetwork::logInProgress(Wait& wait, const Data& data, set set) const
{
    if constexpr (T > none && T <= verbose)
    {
        if (wait.tick() >= 300)
        {
            const std::string name = set == training ? "Training in progress...  " : "Evaluation in progress...";
            const int progress = static_cast<int>(this->index / static_cast<float>(data.sets[set].size) * 100);
            tools::log<T, false>("\rEpoch: ", tools::toConstSizeString(this->epoch, 2), " - ", name,
                                 tools::toConstSizeString(progress, 5), "%",
                                 " - Time: ", tools::toConstSizeString<0>(wait.getDuration(), 3), "s");
        }
    }
}

template <class Archive>
void StraightforwardNeuralNetwork::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<StraightforwardNeuralNetwork, NeuralNetwork>();
    archive& boost::serialization::base_object<NeuralNetwork>(*this);
    archive& this->autoSaveFilePath;
    archive& this->autoSaveWhenBetter;
    archive& this->wantToStopTraining;
    archive& this->index;
    archive& this->isIdle;
    archive& this->epoch;
    archive& this->numberOfTrainingsBetweenTwoEvaluations;
}
}  // namespace snn
