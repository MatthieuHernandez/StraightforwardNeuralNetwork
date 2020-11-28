#pragma once
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>

namespace snn::internal
{
    class LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version) {}

    public:
        LayerOptimizer() = default; // use restricted to Boost library only
        LayerOptimizer(const LayerOptimizer& layer) = default;
        virtual ~LayerOptimizer() = default;

        virtual std::unique_ptr<LayerOptimizer> clone(LayerOptimizer* optimizer) const = 0;

        virtual void applyBefore(std::vector<float>& inputs) = 0;
        virtual void applyAfterForBackpropagation(std::vector<float>& outputs) = 0;

        virtual bool operator==(const LayerOptimizer& optimizer) const = 0;
        virtual bool operator!=(const LayerOptimizer& optimizer) const = 0;
    };
}
