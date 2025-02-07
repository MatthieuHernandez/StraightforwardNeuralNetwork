#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "FilterLayer.hpp"

namespace snn::internal
{
class MaxPooling2D final : public FilterLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        int numberOfOutputs{};
        std::vector<int> maxValueIndexes;

        [[nodiscard]] auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> final;
        [[nodiscard]] auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> final;
        void computeTrain([[maybe_unused]] std::vector<float>& inputErrors) final {}
        void buildKernelIndexes() final;

    public:
        MaxPooling2D() = default;  // use restricted to Boost library only
        explicit MaxPooling2D(LayerModel& model);
        ~MaxPooling2D() final = default;
        MaxPooling2D(const MaxPooling2D&) = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> final;

        [[nodiscard]] auto isValid() const -> errorType final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const BaseLayer& layer) const -> bool final;
        auto operator!=(const BaseLayer& layer) const -> bool final;
};

template <class Archive>
void MaxPooling2D::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<MaxPooling2D, FilterLayer>();
    archive& boost::serialization::base_object<FilterLayer>(*this);
    archive& this->numberOfOutputs;
    archive& this->maxValueIndexes;
}
}  // namespace snn::internal
