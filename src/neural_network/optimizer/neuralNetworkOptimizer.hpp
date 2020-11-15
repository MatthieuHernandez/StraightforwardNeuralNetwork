#pragma once
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
        virtual ~NeuralNetworkOptimizer() = default;

        virtual bool operator==(const NeuralNetworkOptimizer& optimizer) const;
        virtual bool operator!=(const NeuralNetworkOptimizer& optimizer) const;
    };
}
