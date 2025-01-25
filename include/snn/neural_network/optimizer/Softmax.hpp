#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <vector>

#include "LayerOptimizer.hpp"

namespace snn::internal
{
class Softmax final : public LayerOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        static void computeSoftmax(std::vector<float>& outputs);

    public:
        Softmax() = default;  // use restricted to Boost library only
        Softmax(const BaseLayer* layer);
        Softmax(const Softmax& Softmax, const BaseLayer* layer);
        ~Softmax() override = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> override;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) override;
        void applyAfterOutputForTesting(std::vector<float>& outputs) override;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const LayerOptimizer& optimizer) const -> bool override;
        auto operator!=(const LayerOptimizer& optimizer) const -> bool override;
};

template <class Archive>
void Softmax::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<Softmax, LayerOptimizer>();
    ar& boost::serialization::base_object<LayerOptimizer>(*this);
}
}  // namespace snn::internal
