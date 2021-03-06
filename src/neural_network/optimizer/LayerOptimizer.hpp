#pragma once
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include "../layer/BaseLayer.hpp"

namespace snn::internal
{
    class LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        const BaseLayer* layer;

    public:
        LayerOptimizer() = default; // use restricted to Boost library only
        LayerOptimizer(const BaseLayer* layer);
        virtual ~LayerOptimizer() = default;

        virtual std::unique_ptr<LayerOptimizer> clone(const BaseLayer* layer) const = 0;

        virtual void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) = 0;
        virtual void applyAfterOutputForTesting(std::vector<float>& outputs) = 0;

        virtual void applyBeforeBackpropagation(std::vector<float>& inputErrors) = 0;

        virtual bool operator==(const LayerOptimizer& optimizer) const;
        virtual bool operator!=(const LayerOptimizer& optimizer) const;
    };
    
    template <class Archive>
    void LayerOptimizer::serialize(Archive& ar, unsigned version)
    {
        ar & this->layer;
    }
}
