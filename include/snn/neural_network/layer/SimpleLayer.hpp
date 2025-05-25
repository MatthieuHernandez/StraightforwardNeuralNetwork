#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "Layer.hpp"

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
        [[nodiscard]] auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> final;
        [[nodiscard]] auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> final;
        void computeTrain(std::vector<float>& inputErrors) final;

    public:
        SimpleLayer() = default;  // use restricted to Boost library only
        SimpleLayer(SimpleLayer&&) = delete;
        auto operator=(const SimpleLayer&) -> SimpleLayer& = delete;
        auto operator=(SimpleLayer&&) -> SimpleLayer& = delete;
        SimpleLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleLayer(const SimpleLayer&) = default;
        ~SimpleLayer() override = default;

        [[nodiscard]] auto getShapeOfInput() const -> std::vector<int> final;
        [[nodiscard]] auto getShapeOfOutput() const -> std::vector<int> final;
        [[nodiscard]] auto isValid() const -> errorType final;

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

extern template class snn::internal::SimpleLayer<snn::internal::SimpleNeuron>;
extern template class snn::internal::SimpleLayer<snn::internal::RecurrentNeuron>;
extern template class snn::internal::SimpleLayer<snn::internal::GatedRecurrentUnit>;
