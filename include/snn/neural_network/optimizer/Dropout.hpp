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
        void serialize(Archive& archive, uint32_t version);

        float value{};
        float reverseValue{};
        std::uniform_real_distribution<> dist;
        std::vector<bool> presenceProbabilities;

    public:
        Dropout() = default;  // use restricted to Boost library only
        Dropout(float value, const BaseLayer* layer);
        Dropout(const Dropout& dropout, const BaseLayer* layer);
        ~Dropout() final = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> final;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) final;
        void applyAfterOutputForTesting(std::vector<float>& outputs) final;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const LayerOptimizer& optimizer) const -> bool final;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool final;
};

template <class Archive>
void Dropout::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Dropout, LayerOptimizer>();
    archive& boost::serialization::base_object<LayerOptimizer>(*this);
    archive& this->value;
    archive& this->reverseValue;
    archive& this->presenceProbabilities;
}
}  // namespace snn::internal
