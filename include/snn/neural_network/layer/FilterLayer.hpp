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
        void serialize(Archive& ar, unsigned version);

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
        virtual ~FilterLayer() = default;
        FilterLayer(const FilterLayer&) = default;

        [[nodiscard]] std::vector<int> getShapeOfInput() const final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const final;
        [[nodiscard]] int getKernelSize() const;
        [[nodiscard]] auto isValid() const -> ErrorType override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
};

template <class Archive>
void FilterLayer::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<FilterLayer, Layer>();
    ar& boost::serialization::base_object<Layer>(*this);
    ar& this->numberOfFilters;
    ar& this->numberOfKernels;
    ar& this->numberOfKernelsPerFilter;
    ar& this->numberOfNeuronsPerFilter;
    ar& this->kernelSize;
    ar& this->sizeOfNeuronInputs;
    ar& this->shapeOfInput;
    ar& this->shapeOfOutput;
    ar& this->kernelIndexes;
}
}  // namespace snn::internal
