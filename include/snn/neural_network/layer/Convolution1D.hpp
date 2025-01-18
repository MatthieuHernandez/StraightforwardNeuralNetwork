#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "FilterLayer.hpp"

namespace snn::internal
{
class Convolution1D final : public FilterLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) override;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) override;
        void computeTrain(std::vector<float>& inputErrors) override;
        void buildKernelIndexes() override;

    public:
        Convolution1D() = default;  // use restricted to Boost library only
        Convolution1D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~Convolution1D() override = default;
        Convolution1D(const Convolution1D&) = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(
            std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;

        [[nodiscard]] int isValid() const override;

        [[nodiscard]] std::string summary() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
};

template <class Archive>
void Convolution1D::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    boost::serialization::void_cast_register<Convolution1D, FilterLayer>();
    ar& boost::serialization::base_object<FilterLayer>(*this);
}
}  // namespace snn::internal
