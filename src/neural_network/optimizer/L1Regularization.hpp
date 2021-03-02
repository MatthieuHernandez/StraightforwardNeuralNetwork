#pragma once
#include <vector>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "LayerOptimizer.hpp"

namespace snn::internal
{
    class L1Regularization final : public LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float value;

        void applyAfterOutput(std::vector<float>& outputs);

    public:
        L1Regularization() = default;  // use restricted to Boost library only
        L1Regularization(float value, BaseLayer* layer);
        L1Regularization(const L1Regularization& regularization) = default;
        ~L1Regularization() = default;

        std::unique_ptr<LayerOptimizer> clone(LayerOptimizer* optimizer) const override;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) override;
        void applyAfterOutputForTesting(std::vector<float>& outputs) override;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
    };

    template <class Archive>
    void L1Regularization::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<L1Regularization, LayerOptimizer>();
        ar & boost::serialization::base_object<LayerOptimizer>(*this);
        ar & this->value;
    }
}
