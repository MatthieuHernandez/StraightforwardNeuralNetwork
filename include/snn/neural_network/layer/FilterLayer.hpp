#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "Layer.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
class FilterLayer : public Layer<SimpleNeuron>
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    protected:
        int numberOfFilters;
        int numberOfKernels;
        int numberOfKernelsPerFilter;
        int numberOfNeuronsPerFilter;
        int kernelSize;
        int sizeOfNeuronInputs;
        std::vector<int> shapeOfInput;
        std::vector<int> shapeOfOutput;
        vector2D<int> kernelIndexes;

        virtual void buildKernelIndexes() = 0;

    public:
        FilterLayer() = default;  // use restricted to Boost library only
        FilterLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~FilterLayer() override = default;
        FilterLayer(const FilterLayer&) = default;

        [[nodiscard]] auto getShapeOfInput() const -> std::vector<int> final;
        [[nodiscard]] auto getShapeOfOutput() const -> std::vector<int> final;
        [[nodiscard]] auto getKernelSize() const -> int;
        [[nodiscard]] auto isValid() const -> ErrorType override;

        auto operator==(const BaseLayer& layer) const -> bool override;
        auto operator!=(const BaseLayer& layer) const -> bool override;
};

template <class Archive>
void FilterLayer::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<FilterLayer, Layer>();
    archive& boost::serialization::base_object<Layer>(*this);
    archive& this->numberOfFilters;
    archive& this->numberOfKernels;
    archive& this->numberOfKernelsPerFilter;
    archive& this->numberOfNeuronsPerFilter;
    archive& this->kernelSize;
    archive& this->sizeOfNeuronInputs;
    archive& this->shapeOfInput;
    archive& this->shapeOfOutput;
    archive& this->kernelIndexes;
}
}  // namespace snn::internal
