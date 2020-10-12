#pragma once
#include <vector>
#include <boost/serialization/unique_ptr.hpp>
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

        virtual void apply(std::vector<float>& output) = 0;
        virtual void applyForBackpropagation(std::vector<float>& output) = 0;

        bool operator==(const Optimizer& optimizer) const override
        {
            return this->Optimizer::operator==(optimizer);
        }

        bool operator!=(const Optimizer& optimizer) const override
        {
            return this->Optimizer::operator!=(optimizer);
        }
    };

    template <class Archive>
    void LayerOptimizer::serialize(Archive& ar, unsigned version)
    {
        //boost::serialization::void_cast_register<LayerOptimizer, Optimizer>();
        //ar & boost::serialization::base_object<Optimizer>(*this);
    }
}
