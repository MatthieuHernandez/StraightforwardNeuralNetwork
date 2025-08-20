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
        void serialize(Archive& archive, uint32_t version);

        static void computeSoftmax(std::vector<float>& outputs);

    public:
        Softmax() = default;  // use restricted to Boost library only
        explicit Softmax(const BaseLayer* layer);
        Softmax(const Softmax& Softmax, const BaseLayer* layer);
        ~Softmax() final = default;

        auto clone(const BaseLayer* newLayer) const -> std::unique_ptr<LayerOptimizer> final;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) final;
        void applyAfterOutputForTesting(std::vector<float>& outputs) final;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const LayerOptimizer& optimizer) const -> bool final;
};

template <class Archive>
void Softmax::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Softmax, LayerOptimizer>();
    archive& boost::serialization::base_object<LayerOptimizer>(*this);
}
}  // namespace snn::internal
