#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "FilterLayer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"

namespace snn::internal
{
    class LocallyConnected2D final : public FilterLayer
    {
    private :
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        [[nodiscard]] std::vector<float> computeBackOutput(std::vector<float>& inputErrors) override;
        [[nodiscard]] std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) override;
        void computeTrain(std::vector<float>& inputErrors) override;
        void buildKernelIndexes() override;

    public :
        LocallyConnected2D() = default; // use restricted to Boost library only
        LocallyConnected2D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        ~LocallyConnected2D() = default;
        LocallyConnected2D(const LocallyConnected2D&) = default;
        std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;

        [[nodiscard]] int isValid() const override;

        [[nodiscard]] std::string summary() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void LocallyConnected2D::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<LocallyConnected2D, FilterLayer>();
        ar & boost::serialization::base_object<FilterLayer>(*this);
    }
}
