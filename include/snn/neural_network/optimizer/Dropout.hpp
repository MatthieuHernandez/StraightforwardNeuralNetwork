#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <random>
#include <vector>

#include "LayerOptimizer.hpp"

namespace snn::internal
{
class Dropout final : public LayerOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

        float value;
        float reverseValue;
        std::uniform_real_distribution<> dist;
        std::vector<bool> presenceProbabilities;

    public:
        Dropout() = default;  // use restricted to Boost library only
        Dropout(float value, const BaseLayer* layer);
        Dropout(const Dropout& dropout, const BaseLayer* layer);
        ~Dropout() override = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> override;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) override;
        void applyAfterOutputForTesting(std::vector<float>& outputs) override;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const LayerOptimizer& optimizer) const -> bool override;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool override;
};

template <class Archive>
void Dropout::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Dropout, LayerOptimizer>();
    ar& boost::serialization::base_object<LayerOptimizer>(*this);
    ar& this->value;
    ar& this->reverseValue;
    ar& this->presenceProbabilities;
}
}  // namespace snn::internal
