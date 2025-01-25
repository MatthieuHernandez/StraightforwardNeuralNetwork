#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "FilterLayer.hpp"

namespace snn::internal
{
class MaxPooling1D final : public FilterLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, uint32_t version);

        int numberOfOutputs;
        std::vector<int> maxValueIndexes;

        [[nodiscard]] auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> override;
        [[nodiscard]] auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> override;
        void computeTrain([[maybe_unused]] std::vector<float>& inputErrors) override {}
        void buildKernelIndexes() override;

    public:
        MaxPooling1D() = default;  // use restricted to Boost library only
        MaxPooling1D(LayerModel& model);
        ~MaxPooling1D() override = default;
        MaxPooling1D(const MaxPooling1D&) = default;
        auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer> override;

        [[nodiscard]] auto isValid() const -> ErrorType override;

        [[nodiscard]] auto summary() const -> std::string override;

        auto operator==(const BaseLayer& layer) const -> bool override;
        auto operator!=(const BaseLayer& layer) const -> bool override;
};

template <class Archive>
void MaxPooling1D::serialize(Archive& ar, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<MaxPooling1D, FilterLayer>();
    ar& boost::serialization::base_object<FilterLayer>(*this);
    ar& this->numberOfOutputs;
    ar& this->maxValueIndexes;
}
}  // namespace snn::internal
