#pragma once

namespace snn::internal
{

    class Optimizer
    {

    };

    class StochasticGradientDescent //: public Optimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        float learningRate = 0.03f;
        float momentum = 0.0f;

        StochasticGradientDescent() = default;
        StochasticGradientDescent(const StochasticGradientDescent& sgd) = default;
        ~StochasticGradientDescent() = default;

        bool operator==(const StochasticGradientDescent& sgd) const 
        {
            return this->learningRate == sgd.learningRate
                && this->momentum == sgd.momentum;
        }
        bool operator!=(const StochasticGradientDescent& sgd) const 
        { 
            return !(*this == sgd); 
        }
    };

    template <class Archive>
    void StochasticGradientDescent::serialize(Archive& ar, const unsigned int version)
    {
        ar & this->learningRate;
        ar & this->momentum;
    }
}