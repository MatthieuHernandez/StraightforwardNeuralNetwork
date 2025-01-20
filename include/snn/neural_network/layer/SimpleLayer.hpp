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
        void serialize(Archive& ar, unsigned version);

    protected:
        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) final;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) final;
        void computeTrain(std::vector<float>& inputErrors) final;

    public:
        SimpleLayer() = default;  // use restricted to Boost library only
        SimpleLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleLayer(const SimpleLayer&) = default;
        virtual ~SimpleLayer() = default;

        [[nodiscard]] std::vector<int> getShapeOfInput() const final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const final;
        [[nodiscard]] auto isValid() const -> ErrorType final;

        bool operator==(const BaseLayer& layer) const final;
        bool operator!=(const BaseLayer& layer) const final;
};

template <BaseNeuron N>
template <class Archive>
void SimpleLayer<N>::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<SimpleLayer<N>, Layer<N>>();
    ar& boost::serialization::base_object<Layer<N>>(*this);
}

template <>
std::vector<float> SimpleLayer<RecurrentNeuron>::computeOutput(const std::vector<float>& inputs, bool temporalReset);

template <>
std::vector<float> SimpleLayer<GatedRecurrentUnit>::computeOutput(const std::vector<float>& inputs, bool temporalReset);

#include "SimpleLayer.tpp"
}  // namespace snn::internal
