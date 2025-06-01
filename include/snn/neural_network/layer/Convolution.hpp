#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <memory>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "FilterLayer.hpp"

namespace snn::internal
{
class Convolution : public FilterLayer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        [[nodiscard]] auto computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float> final;
        [[nodiscard]] auto computeOutput(const std::vector<float>& inputs, bool temporalReset)
            -> std::vector<float> final;
        void computeTrain(std::vector<float>& inputErrors) final;

    public:
        Convolution() = default;  // use restricted to Boost library only
        Convolution(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~Convolution() override = default;
        Convolution(const Convolution&) = default;
        Convolution(Convolution&&) = delete;
        auto operator=(const Convolution&) -> Convolution& = delete;
        auto operator=(Convolution&&) -> Convolution& = delete;
};

template <class Archive>
void Convolution::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Convolution, FilterLayer>();
    archive& boost::serialization::base_object<FilterLayer>(*this);
}
}  // namespace snn::internal
