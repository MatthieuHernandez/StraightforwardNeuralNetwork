#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "FilterLayer.hpp"

namespace snn::internal
{
class LocallyConnected2D final : public FilterLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        [[nodiscard]] auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> final;
        [[nodiscard]] auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> final;
        void computeTrain(std::vector<float>& inputErrors) final;
        void buildKernelIndexes() final;

    public:
        LocallyConnected2D() = default;  // use restricted to Boost library only
        LocallyConnected2D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~LocallyConnected2D() final = default;
        LocallyConnected2D(const LocallyConnected2D&) = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> final;

        [[nodiscard]] auto isValid() const -> errorType final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const BaseLayer& layer) const -> bool final;
};

template <class Archive>
void LocallyConnected2D::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<LocallyConnected2D, FilterLayer>();
    archive& boost::serialization::base_object<FilterLayer>(*this);
}
}  // namespace snn::internal
