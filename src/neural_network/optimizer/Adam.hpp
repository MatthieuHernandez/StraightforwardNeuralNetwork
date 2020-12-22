#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "NeuralNetworkOptimizer.hpp"


namespace snn::internal
{
    class Adam final : public NeuralNetworkOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        float learningRate;
        float beta1;
        float beta2;
        float epsilon;

        float reverseBeta1;
        float reverseBeta2;
        float precomputedM;
        float precomputedV;
        float precomputedDelta;

        Adam() = default;
        Adam(float learningRate, float beta1, float beta2, float epsilon);
        Adam(const Adam& sgd) = default;
        ~Adam() override = default;
        [[nodiscard]] std::shared_ptr<NeuralNetworkOptimizer> clone() const override;

        void updateWeights(SimpleNeuron& neuron, float error) const override;
        void updateWeights(RecurrentNeuron& neuron, float error) const override;

        [[nodiscard]] int isValid() override;

        void operator++() override;
        bool operator==(const NeuralNetworkOptimizer& optimizer) const override;
        bool operator!=(const NeuralNetworkOptimizer& optimizer) const override;
    };

    template <class Archive>
    void Adam::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<Adam, NeuralNetworkOptimizer>();
        ar & boost::serialization::base_object<NeuralNetworkOptimizer>(*this);
        ar & this->learningRate;
        ar & this->beta1;
        ar & this->beta2;
        ar & this->epsilon;
    }
}
