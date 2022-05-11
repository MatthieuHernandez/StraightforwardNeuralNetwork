#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "neuron/SimpleNeuron.hpp"

namespace snn::internal
{
    class FilterLayer : public Layer<SimpleNeuron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected :
        int numberOfFilters;
        int numberOfKernels;
        float numberOfKernelsPerFilter;
        int kernelSize;
        std::vector<int> shapeOfInput;
        std::vector<int> shapeOfOutput;

        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) override = 0;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) override = 0;

    public:
        FilterLayer() = default;  // use restricted to Boost library only
        FilterLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        virtual ~FilterLayer() = default;
        FilterLayer(const FilterLayer&) = default;


        [[nodiscard]] std::vector<int> getShapeOfInput() const override final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override final;
        [[nodiscard]] int getKernelSize() const;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void FilterLayer::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<FilterLayer, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
        ar & this->numberOfFilters;
        ar & this->numberOfKernels;
        ar & this->numberOfKernelsPerFilter;
        ar & this->kernelSize;
        ar & this->shapeOfInput;
        ar & this->shapeOfOutput;
    }
}
