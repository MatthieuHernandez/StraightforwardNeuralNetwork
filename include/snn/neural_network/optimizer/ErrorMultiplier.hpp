#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <vector>

#include "LayerOptimizer.hpp"

namespace snn::internal
{
class ErrorMultiplier final : public LayerOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float factor;

    public:
        ErrorMultiplier() = default;  // use restricted to Boost library only
        ErrorMultiplier(float value, BaseLayer* layer);
        ErrorMultiplier(const ErrorMultiplier& errorMultiplier, const BaseLayer* layer);
        ~ErrorMultiplier() override = default;

        std::unique_ptr<LayerOptimizer> clone(const BaseLayer* newLayer) const override;

        void applyAfterOutputForTraining(std::vector<float>&, bool) override {}
        void applyAfterOutputForTesting(std::vector<float>&) override {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::string summary() const override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
};

template <class Archive>
void ErrorMultiplier::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<ErrorMultiplier, LayerOptimizer>();
    ar& boost::serialization::base_object<LayerOptimizer>(*this);
    ar& this->factor;
}
}  // namespace snn::internal
