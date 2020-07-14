#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"

namespace snn::internal
{
    //extern template class Layer<Perceptron>;

    class Filter : public Layer<Perceptron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected :
        int numberOfFilters;
        int sizeOfFilterMatrix;
        std::vector<int> shapeOfInput;

        [[nodiscard]] virtual std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const = 0;
        virtual void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const = 0;

    public:
        Filter() = default;  // use restricted to Boost library only
        Filter(LayerModel& model, StochasticGradientDescent* optimizer);
        ~Filter() = default;
        Filter(const Filter&) = default;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override final;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override final;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override = 0;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void Filter::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Filter, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
        ar & this->numberOfFilters;
        ar & this->sizeOfFilterMatrix;
        ar & this->shapeOfInput;
    }
}