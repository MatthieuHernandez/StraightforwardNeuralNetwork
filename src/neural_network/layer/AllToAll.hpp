#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"
#include "perceptron/activation_function/ActivationFunction.hpp"

namespace snn::internal
{
    class AllToAll final : public Layer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const override;
        void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const override;

    public:
        AllToAll() = default;  // use restricted to Boost library only
        AllToAll(LayerModel& model, StochasticGradientDescent* optimizer);
        AllToAll(const AllToAll&) = default;
        ~AllToAll() = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const AllToAll& layer) const;
        bool operator!=(const AllToAll& layer) const;
    };

    template <class Archive>
    void AllToAll::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<AllToAll, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
    }
}
