#pragma once
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "Optimizer.hpp"

namespace snn::internal
{
    class LayerOptimizer : public Optimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        LayerOptimizer() = default; // use restricted to Boost library only
        LayerOptimizer(const LayerOptimizer& layer) = default;
        virtual ~LayerOptimizer() = default;

        virtual std::unique_ptr<LayerOptimizer> clone(LayerOptimizer* optimizer) const = 0;

        virtual void applyBefore(std::vector<float>& inputs) = 0;
        virtual void applyAfterForBackpropagation(std::vector<float>& outputs) = 0;

        bool operator==(const Optimizer& optimizer) const override;
        bool operator!=(const Optimizer& optimizer) const override;
    };

    template <class Archive>
    void LayerOptimizer::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<LayerOptimizer, Optimizer>();
        ar & boost::serialization::base_object<Optimizer>(*this);
    }
}
