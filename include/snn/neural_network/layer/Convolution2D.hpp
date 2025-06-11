#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "Convolution.hpp"

namespace snn::internal
{
class Convolution2D final : public Convolution
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        void buildKernelIndexes() final;

    public:
        Convolution2D() = default;  // use restricted to Boost library only
        Convolution2D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~Convolution2D() final = default;
        Convolution2D(const Convolution2D&) = default;
        [[nodiscard]] auto clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
            -> std::unique_ptr<BaseLayer> final;

        [[nodiscard]] auto isValid() const -> errorType final;

        [[nodiscard]] auto summary() const -> std::string final;

        auto operator==(const BaseLayer& layer) const -> bool final;
        auto operator!=(const BaseLayer& layer) const -> bool final;
};

template <class Archive>
void Convolution2D::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<Convolution2D, Convolution>();
    archive& boost::serialization::base_object<Convolution>(*this);
}
}  // namespace snn::internal
