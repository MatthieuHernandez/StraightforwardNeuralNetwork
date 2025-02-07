#pragma once
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <memory>

#include "StatisticAnalysis.hpp"
#include "layer/BaseLayer.hpp"
#include "layer/Convolution1D.hpp"
#include "layer/Convolution2D.hpp"
#include "layer/FullyConnected.hpp"
#include "layer/GruLayer.hpp"
#include "layer/LayerModel.hpp"
#include "layer/LocallyConnected1D.hpp"
#include "layer/LocallyConnected2D.hpp"
#include "layer/MaxPooling1D.hpp"
#include "layer/MaxPooling2D.hpp"
#include "layer/Recurrence.hpp"
#include "optimizer/NeuralNetworkOptimizer.hpp"
#include "optimizer/NeuralNetworkOptimizerModel.hpp"

namespace snn::internal
{
class NeuralNetwork : public StatisticAnalysis
{
    private:
        bool outputNan = false;
        int64_t numberOfTraining = 0;

        void backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired,
                                      bool temporalReset);
        [[nodiscard]] auto calculateError(const std::vector<float>& outputs, const std::vector<float>& desired) const
            -> std::vector<float>;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    protected:
        auto output(const std::vector<float>& inputs, bool temporalReset) -> std::vector<float>;
        auto outputForTraining(const std::vector<float>& inputs,
                               bool temporalReset)
            -> std::vector<float>;  // Because Dropout is different during training and inference

        void evaluateOnceForRegression(const std::vector<float>& inputs, const std::vector<float>& desired,
                                       float precision, bool temporalReset);
        void evaluateOnceForMultipleClassification(const std::vector<float>& inputs, const std::vector<float>& desired,
                                                   float separator, bool temporalReset);
        void evaluateOnceForClassification(const std::vector<float>& inputs, int classNumber, float separator,
                                           bool temporalReset);

        auto getLayerOutputs(const std::vector<float>& inputs) -> vector2D<float>;

    public:
        NeuralNetwork() = default;  // use restricted to Boost library only
        NeuralNetwork(NeuralNetwork&&) = delete;
        auto operator=(const NeuralNetwork&) -> NeuralNetwork& = default;
        auto operator=(NeuralNetwork&&) -> NeuralNetwork& = delete;
        NeuralNetwork(std::vector<LayerModel>& architecture, NeuralNetworkOptimizerModel optimizer);
        NeuralNetwork(const NeuralNetwork& neuralNetwork);
        ~NeuralNetwork() override = default;

        [[nodiscard]] auto hasNan() const -> bool;
        [[nodiscard]] auto getNumberOfTraining() const -> int64_t;
        [[nodiscard]] auto getNumberOfLayers() const -> int;
        [[nodiscard]] auto getNumberOfInputs() const -> int;
        [[nodiscard]] auto getNumberOfOutputs() const -> int;
        [[nodiscard]] auto getNumberOfNeurons() const -> int;
        [[nodiscard]] auto getNumberOfParameters() const -> int;

        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        std::vector<std::unique_ptr<BaseLayer>> layers;

        [[nodiscard]] auto isValid() const -> errorType;

        void trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired, bool temporalReset = true);

        auto operator==(const NeuralNetwork& neuralNetwork) const -> bool;
        auto operator!=(const NeuralNetwork& neuralNetwork) const -> bool;
};

template <class Archive>
void NeuralNetwork::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
    archive& boost::serialization::base_object<StatisticAnalysis>(*this);
    archive.template register_type<FullyConnected>();
    archive.template register_type<Recurrence>();
    archive.template register_type<GruLayer>();
    archive.template register_type<Convolution1D>();
    archive.template register_type<Convolution2D>();
    archive.template register_type<LocallyConnected1D>();
    archive.template register_type<LocallyConnected2D>();
    archive.template register_type<StochasticGradientDescent>();
    archive.template register_type<MaxPooling1D>();
    archive.template register_type<MaxPooling2D>();
    if (version >= 1)
    {
        archive& this->numberOfTraining;
    }
    archive& this->layers;
    archive& this->optimizer;
}
}  // namespace snn::internal
BOOST_CLASS_VERSION(snn::internal::NeuralNetwork, 1)
