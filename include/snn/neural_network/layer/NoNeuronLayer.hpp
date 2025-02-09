#pragma once
#include <boost/serialization/access.hpp>
#include <memory>

#include "BaseLayer.hpp"
#include "LayerModel.hpp"

namespace snn::internal
{
class NoNeuronLayer : public BaseLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    protected:
        int numberOfInputs{};
        int numberOfOutputs{};

    public:
        NoNeuronLayer() = default;  // use restricted to Boost library only
        explicit NoNeuronLayer(LayerModel& model)
            : numberOfInputs(model.numberOfInputs),
              numberOfOutputs(model.numberOfOutputs)
        {
        }
        ~NoNeuronLayer() override = default;

        [[nodiscard]] auto getNumberOfOutput() const -> int;
        [[nodiscard]] auto getAverageOfAbsNeuronWeights() const -> float final;
        [[nodiscard]] auto getAverageOfSquareNeuronWeights() const -> float final;
        [[nodiscard]] auto getNeuron(int index) -> void* final;
        [[nodiscard]] auto getNumberOfNeurons() const -> int final;
        [[nodiscard]] auto getNumberOfParameters() const -> int final;
};

template <class Archive>
void NoNeuronLayer::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<NoNeuronLayer, BaseLayer>();
    archive& boost::serialization::base_object<BaseLayer>(*this);
    archive& this->numberOfInputs;
    archive& this->numberOfOutputs;
}
}  // namespace snn::internal
