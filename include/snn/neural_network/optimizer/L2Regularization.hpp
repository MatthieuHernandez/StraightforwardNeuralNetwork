#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <sstream>
#include <vector>

#include "LayerOptimizer.hpp"

namespace snn::internal
{
class L2Regularization final : public LayerOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

        float value;

    public:
        L2Regularization() = default;  // use restricted to Boost library only
        L2Regularization(float value, BaseLayer* layer);
        L2Regularization(const L2Regularization& regularization, const BaseLayer* layer);
        ~L2Regularization() override = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> override;

        void applyAfterOutputForTraining(std::vector<float>&, bool) override {}
        void applyAfterOutputForTesting(std::vector<float>&) override {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const LayerOptimizer& optimizer) const -> bool override;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool override;
};

template <class Archive>
void L2Regularization::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<L2Regularization, LayerOptimizer>();
    ar& boost::serialization::base_object<LayerOptimizer>(*this);
    ar& this->value;
}
}  // namespace snn::internal
