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
        void serialize(Archive& ar, uint32_t version);

    protected:
        int numberOfInputs;
        int numberOfOutputs;

    public:
        NoNeuronLayer() = default;  // use restricted to Boost library only
        NoNeuronLayer(LayerModel& model)
        {
            this->numberOfInputs = model.numberOfInputs;
            this->numberOfOutputs = model.numberOfOutputs;
        }
        virtual ~NoNeuronLayer() = default;

        [[nodiscard]] auto getNumberOfOutput() const -> int;
        [[nodiscard]] auto getAverageOfAbsNeuronWeights() const -> float override;
        [[nodiscard]] auto getAverageOfSquareNeuronWeights() const -> float override;
        [[nodiscard]] auto getNeuron(int index) -> void* override;
        [[nodiscard]] auto getNumberOfNeurons() const -> int override;
        [[nodiscard]] auto getNumberOfParameters() const -> int override;
};

template <class Archive>
void NoNeuronLayer::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<NoNeuronLayer, BaseLayer>();
    ar& boost::serialization::base_object<BaseLayer>(*this);
    ar& this->numberOfInputs;
    ar& this->numberOfOutputs;
}
}  // namespace snn::internal
