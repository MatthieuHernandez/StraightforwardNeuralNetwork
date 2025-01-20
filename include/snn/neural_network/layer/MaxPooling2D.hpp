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
        void serialize(Archive& ar, unsigned version);

        int numberOfOutputs;
        std::vector<int> maxValueIndexes;

        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) override;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) override;
        void computeTrain([[maybe_unused]] std::vector<float>& inputErrors) override {}
        void buildKernelIndexes() override;

    public:
        MaxPooling2D() = default;  // use restricted to Boost library only
        MaxPooling2D(LayerModel& model);
        ~MaxPooling2D() override = default;
        MaxPooling2D(const MaxPooling2D&) = default;
        std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;

        [[nodiscard]] auto isValid() const -> ErrorType override;

        [[nodiscard]] std::string summary() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
};

template <class Archive>
void MaxPooling2D::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<MaxPooling2D, FilterLayer>();
    ar& boost::serialization::base_object<FilterLayer>(*this);
    ar& this->numberOfOutputs;
    ar& this->maxValueIndexes;
}
}  // namespace snn::internal
