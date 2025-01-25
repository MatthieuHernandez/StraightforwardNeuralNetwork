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
        void serialize(Archive& archive, uint32_t version);

        float factor;

    public:
        ErrorMultiplier() = default;  // use restricted to Boost library only
        ErrorMultiplier(float value, BaseLayer* layer);
        ErrorMultiplier(const ErrorMultiplier& errorMultiplier, const BaseLayer* layer);
        ~ErrorMultiplier() final = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> final;

        void applyAfterOutputForTraining(std::vector<float>&, bool) final {}
        void applyAfterOutputForTesting(std::vector<float>&) final {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const LayerOptimizer& optimizer) const -> bool final;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool final;
};

template <class Archive>
void ErrorMultiplier::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<ErrorMultiplier, LayerOptimizer>();
    archive& boost::serialization::base_object<LayerOptimizer>(*this);
    archive& this->factor;
}
}  // namespace snn::internal
