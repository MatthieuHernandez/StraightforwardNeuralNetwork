#pragma once
#include <vector>
#include <sstream>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "LayerOptimizer.hpp"

namespace snn::internal
{
    class L2Regularization final : public LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float value;

    public:
        L2Regularization() = default;  // use restricted to Boost library only
        L2Regularization(float value, BaseLayer* layer);
        L2Regularization(const L2Regularization& regularization, const BaseLayer* layer);
        ~L2Regularization() override = default;

        std::unique_ptr<LayerOptimizer> clone(const BaseLayer* newLayer) const override;

        void applyAfterOutputForTraining(std::vector<float>&, bool) override {}
        void applyAfterOutputForTesting(std::vector<float>&) override {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::string summary() const override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
    };

    template <class Archive>
    void L2Regularization::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<L2Regularization, LayerOptimizer>();
        ar & boost::serialization::base_object<LayerOptimizer>(*this);
        ar & this->value;
    }
}
