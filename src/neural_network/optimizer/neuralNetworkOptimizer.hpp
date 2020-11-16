#pragma once
#include <memory>
#include <boost/serialization/access.hpp>

namespace snn
{
    class NeuralNetworkOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version) {}

    public:
        NeuralNetworkOptimizer() = default;
        virtual ~NeuralNetworkOptimizer() = default;
        virtual std::shared_ptr<NeuralNetworkOptimizer> clone() const = 0;

        virtual void updateWeight(const float& error, float& weight, float& previousDeltaWeight, const float& lastInput) const = 0;

        [[nodiscard]] virtual int isValid() = 0;

        virtual bool operator==(const NeuralNetworkOptimizer& optimizer) const;
        virtual bool operator!=(const NeuralNetworkOptimizer& optimizer) const;
    };
}
