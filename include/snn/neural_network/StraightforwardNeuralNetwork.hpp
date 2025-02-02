#pragma once
#include <string>
#include <thread>
#include <vector>

#include "../data/Dataset.hpp"
#include "../tools/Error.hpp"
#include "NeuralNetwork.hpp"
#include "Wait.hpp"
#include "layer/LayerFactory.hpp"  // IWYU pragma: keep
#include "layer/LayerModel.hpp"
#include "optimizer/LayerOptimizerFactory.hpp"          // IWYU pragma: keep
#include "optimizer/NeuralNetworkOptimizerFactory.hpp"  // IWYU pragma: keep

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

        void trainSync(Dataset& dataset, Wait wait, int batchSize, int evaluationFrequency);
        void saveSync(const std::string& filePath);
        void evaluate(const Dataset& dataset, Wait& wait);
        void evaluateOnce(const Dataset& dataset);

        auto continueTraining(Wait wait) const -> bool;
        void validData(const Dataset& dataset, int batchSize) const;

        template <logLevel T>
        void logAccuracy(Wait& wait, bool hasSaved) const;
        template <logLevel T>
        void logInProgress(Wait& wait, const Dataset& dataset, setType set) const;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    public:
        StraightforwardNeuralNetwork() = default;  // use restricted to Boost library only
        explicit StraightforwardNeuralNetwork(std::vector<LayerModel> architecture,
                                              NeuralNetworkOptimizerModel optimizer = {
                                                  .type = neuralNetworkOptimizerType::stochasticGradientDescent,
                                                  .learningRate = 0.03F,
                                                  .momentum = 0.0F});
        StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork);
        ~StraightforwardNeuralNetwork() final;

        bool autoSaveWhenBetter = false;
        std::string autoSaveFilePath = "AutoSave.snn";

        [[nodiscard]] auto isValid() const -> errorType;

        void startTrainingAsync(Dataset& dataset, int batchSize = 1, int evaluationFrequency = 1);
        void stopTrainingAsync();

        void waitFor(Wait wait) const;
        void train(Dataset& dataset, Wait wait, int batchSize = 1, int evaluationFrequency = 1);

        void evaluate(const Dataset& dataset);

        auto computeOutput(const std::vector<float>& inputs, bool temporalReset = false) -> std::vector<float>;
        auto computeCluster(const std::vector<float>& inputs, bool temporalReset = false) -> int;

        auto isTraining() const -> bool;

        void saveAs(const std::string& filePath);
        void saveFeatureMapsAsBitmap(const std::string& filePath);
        void saveData2DAsBitmap(const std::string& filePath, const Dataset& dataset, int dataIndex);
        void saveFilterLayersAsBitmap(const std::string& filePath, const Dataset& dataset, int dataIndex);
        static auto loadFrom(const std::string& filePath) -> StraightforwardNeuralNetwork&;

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
    if constexpr (T > none && T <= verbose)  // NOLINT(misc-redundant-expression)
    {
        tools::log<T, false>("\rEpoch: ", tools::toConstSizeString(this->epoch, 2),
                             " - Accuracy: ", tools::toConstSizeString<2>(this->getGlobalClusteringRate(), 4),
                             " - MAE: ", tools::toConstSizeString<4>(this->getMeanAbsoluteError(), 9),
                             " - Time: ", tools::toConstSizeString<0>(wait.getDurationAndReset(), 3), "s");
        if (hasSaved)
        {
            tools::log<T, false>(" - Saved");
        }
        tools::log<T>();
    }
}

template <logLevel T>
void StraightforwardNeuralNetwork::logInProgress(Wait& wait, const Dataset& dataset, setType set) const
{
    if constexpr (T > none && T <= verbose)  // NOLINT(misc-redundant-expression)
    {
        const int refreshRate = 300;
        if (wait.tick() >= refreshRate)
        {
            const std::string name =
                set == setType::training ? "Training in progress...  " : "Evaluation in progress...";
            const size_t size = set == setType::training ? dataset.data.training.size : dataset.data.testing.size;
            const int progress = this->index * 100 / static_cast<int>(size);
            tools::log<T, false>("\rEpoch: ", tools::toConstSizeString(this->epoch, 2), " - ", name,
                                 tools::toConstSizeString(progress, 5), "%",  // NOLINT(*magic-numbers)
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
