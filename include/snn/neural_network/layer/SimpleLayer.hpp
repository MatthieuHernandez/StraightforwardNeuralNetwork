#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/LayerOptimizer.hpp"
#include "Layer.hpp"
#include "neuron/GatedRecurrentUnit.hpp"
#include "neuron/RecurrentNeuron.hpp"

namespace snn::internal
{
template <BaseNeuron N>
class SimpleLayer : public Layer<N>
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    protected:
        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) final;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) final;
        void computeTrain(std::vector<float>& inputErrors) final;

    public:
        SimpleLayer() = default;  // use restricted to Boost library only
        SimpleLayer(SimpleLayer&&) = delete;
        auto operator=(const SimpleLayer&) -> SimpleLayer& = delete;
        auto operator=(SimpleLayer&&) -> SimpleLayer& = delete;
        SimpleLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleLayer(const SimpleLayer&) = default;
        ~SimpleLayer() override = default;

        [[nodiscard]] std::vector<int> getShapeOfInput() const final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const final;
        [[nodiscard]] auto isValid() const -> ErrorType final;

        auto operator==(const BaseLayer& layer) const -> bool override;
        auto operator!=(const BaseLayer& layer) const -> bool override;
};

template <BaseNeuron N>
template <class Archive>
void SimpleLayer<N>::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<SimpleLayer<N>, Layer<N>>();
    archive& boost::serialization::base_object<Layer<N>>(*this);
}

template <>
std::vector<float> SimpleLayer<RecurrentNeuron>::computeOutput(const std::vector<float>& inputs, bool temporalReset);

template <>
std::vector<float> SimpleLayer<GatedRecurrentUnit>::computeOutput(const std::vector<float>& inputs, bool temporalReset);

}  // namespace snn::internal
#include "SimpleLayer.tpp"  // IWYU pragma: keep