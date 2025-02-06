#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <sstream>
#include <vector>

#include "LayerOptimizer.hpp"

namespace snn::internal
{
class L1Regularization final : public LayerOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        float value{};

        void applyAfterOutput(std::vector<float>& outputs);

    public:
        L1Regularization() = default;  // use restricted to Boost library only
        L1Regularization(float value, BaseLayer* layer);
        L1Regularization(const L1Regularization& regularization, const BaseLayer* layer);
        ~L1Regularization() final = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> final;

        void applyAfterOutputForTraining(std::vector<float>&, bool) final {}
        void applyAfterOutputForTesting(std::vector<float>&) final {}

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const LayerOptimizer& optimizer) const -> bool final;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool final;
};

template <class Archive>
void L1Regularization::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<L1Regularization, LayerOptimizer>();
    archive& boost::serialization::base_object<LayerOptimizer>(*this);
    archive& this->value;
}
}  // namespace snn::internal
