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
        void serialize(Archive& archive, uint32_t version);

        float value;

    public:
        L2Regularization() = default;  // use restricted to Boost library only
        L2Regularization(float value, BaseLayer* layer);
        L2Regularization(const L2Regularization& regularization, const BaseLayer* layer);
        ~L2Regularization() final = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> final;

        void applyAfterOutputForTraining(std::vector<float>&, bool) final {}
        void applyAfterOutputForTesting(std::vector<float>&) final {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const LayerOptimizer& optimizer) const -> bool final;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool final;
};

template <class Archive>
void L2Regularization::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<L2Regularization, LayerOptimizer>();
    archive& boost::serialization::base_object<LayerOptimizer>(*this);
    archive& this->value;
}
}  // namespace snn::internal
