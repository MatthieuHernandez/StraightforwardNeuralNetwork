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
        int sizeOfFilterMatrix;
        std::vector<int> shapeOfInput;
        std::vector<int> shapeOfOutput;

        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) override final;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) override final;
        [[nodiscard]] virtual std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) = 0;
        virtual void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) = 0;

    public:
        FilterLayer() = default;  // use restricted to Boost library only
        FilterLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        virtual ~FilterLayer() = default;
        FilterLayer(const FilterLayer&) = default;


        [[nodiscard]] std::vector<int> getShapeOfInput() const override final;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override final;
        [[nodiscard]] int getSizeOfFilterMatrix() const;
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
        ar & this->sizeOfFilterMatrix;
        ar & this->shapeOfInput;
        ar & this->shapeOfOutput;
    }
}
