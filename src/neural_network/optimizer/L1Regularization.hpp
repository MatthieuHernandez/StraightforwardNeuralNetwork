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
        float reverseValue;

    public:
        L1Regularization() = default;  // use restricted to Boost library only
        L1Regularization(float value);
        L1Regularization(const L1Regularization& dropout) = default;
        ~L1Regularization() = default;

        std::unique_ptr<LayerOptimizer> clone(LayerOptimizer* optimizer) const override;

        void applyBefore(std::vector<float>& inputs) override;
        void applyAfterForBackpropagation(std::vector<float>& outputs) override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
    };

    template <class Archive>
    void L1Regularization::serialize(Archive& ar, unsigned version)
    {
        boost::serialization::void_cast_register<L1Regularization, LayerOptimizer>();
        ar & boost::serialization::base_object<LayerOptimizer>(*this);
        ar & this->value;
        ar & this->reverseValue;
    }
}
