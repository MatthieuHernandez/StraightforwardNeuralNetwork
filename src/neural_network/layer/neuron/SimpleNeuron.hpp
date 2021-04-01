#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Neuron.hpp"
#include "input/NeuronInput.hpp"

namespace snn::internal
{
    class SimpleNeuron final : public Neuron
    {
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        SimpleNeuron() = default; // use restricted to Boost library only
        SimpleNeuron(NeuronModel model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        SimpleNeuron(const SimpleNeuron& neuron) = default;
        ~SimpleNeuron() = default;

        template <NeuronInput I>
        [[nodiscard]] float output(const I& inputs);

        [[nodiscard]] std::vector<float>& backOutput(float error);

        void train(float error);

        [[nodiscard]] int isValid() const;

        bool operator==(const SimpleNeuron& neuron) const;
        bool operator!=(const SimpleNeuron& neuron) const;
    };

    template <class Archive>
    void SimpleNeuron::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<SimpleNeuron, Neuron>();
        ar & boost::serialization::base_object<Neuron>(*this);
    }

    template <NeuronInput I>
    float SimpleNeuron::output(const I& inputs)
    {
        this->lastInputs = inputs;
        float tmp = 0.0f; // to activate the SIMD optimization
        #pragma omp simd
        for (size_t w = 0; w < this->weights.size(); ++w)
        {
            tmp += inputs[w] * weights[w];
        }
        this->sum = tmp + bias;
        return this->outputFunction->function(this->sum);
    }
}
