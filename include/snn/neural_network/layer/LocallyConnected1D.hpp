#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "FilterLayer.hpp"

namespace snn::internal
{
class LocallyConnected1D final : public FilterLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        [[nodiscard]] auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> override;
        [[nodiscard]] auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> override;
        void computeTrain(std::vector<float>& inputErrors) override;
        void buildKernelIndexes() override;

    public:
        LocallyConnected1D() = default;  // use restricted to Boost library only
        LocallyConnected1D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~LocallyConnected1D() = default;
        LocallyConnected1D(const LocallyConnected1D&) = default;
        auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer> override;

        [[nodiscard]] auto isValid() const -> ErrorType override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const BaseLayer& layer) const -> bool override;
        auto operator!=(const BaseLayer& layer) const -> bool override;
};

template <class Archive>
void LocallyConnected1D::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<LocallyConnected1D, FilterLayer>();
    ar& boost::serialization::base_object<FilterLayer>(*this);
}
}  // namespace snn::internal
