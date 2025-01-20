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

        std::unique_ptr<LayerOptimizer> clone(const BaseLayer* newLayer) const override;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) override;
        void applyAfterOutputForTesting(std::vector<float>& outputs) override;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::string summary() const override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
};

template <class Archive>
void Softmax::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<Softmax, LayerOptimizer>();
    ar& boost::serialization::base_object<LayerOptimizer>(*this);
}
}  // namespace snn::internal
