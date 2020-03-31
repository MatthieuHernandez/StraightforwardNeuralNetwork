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
        createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const;

    public:
        AllToAll() = default;  // use restricted to Boost library only
        AllToAll(int numberOfInputs,
                 int numberOfNeurons,
                 activationFunction activation,
                 StochasticGradientDescent* optimizer);
        ~AllToAll() = default;
        AllToAll(const AllToAll&) = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        std::vector<float> output(const std::vector<float>& inputs) override;
        std::vector<float> backOutput(std::vector<float>& inputsError) override;
        void train(std::vector<float>& inputsError) override;

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
