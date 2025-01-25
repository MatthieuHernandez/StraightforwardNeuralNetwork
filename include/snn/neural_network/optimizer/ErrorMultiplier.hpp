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
        void serialize(Archive& ar, uint32_t version);

        float factor;

    public:
        ErrorMultiplier() = default;  // use restricted to Boost library only
        ErrorMultiplier(float value, BaseLayer* layer);
        ErrorMultiplier(const ErrorMultiplier& errorMultiplier, const BaseLayer* layer);
        ~ErrorMultiplier() override = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> override;

        void applyAfterOutputForTraining(std::vector<float>&, bool) override {}
        void applyAfterOutputForTesting(std::vector<float>&) override {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const LayerOptimizer& optimizer) const -> bool override;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool override;
};

template <class Archive>
void ErrorMultiplier::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<ErrorMultiplier, LayerOptimizer>();
    ar& boost::serialization::base_object<LayerOptimizer>(*this);
    ar& this->factor;
}
}  // namespace snn::internal
