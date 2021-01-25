#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "BaseLayer.hpp"

namespace snn::internal
{
    class MaxPooling1D final : public BaseLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int sizeOfFilterMatrix;
        std::vector<int> shapeOfInput;

    public:
        MaxPooling1D() = default; // use restricted to Boost library only
        MaxPooling1D(LayerModel& model);
        ~MaxPooling1D() = default;
        MaxPooling1D(const MaxPooling1D&) = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone() const;

        [[nodiscard]] std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override;
        [[nodiscard]] std::vector<float> outputForBackpropagation(const std::vector<float>& inputs, bool temporalReset) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const;

        bool operator==(const BaseLayer& layer) const;
        bool operator!=(const BaseLayer& layer) const;
    };

    template <class Archive>
    void MaxPooling1D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<MaxPooling1D, BaseLayer>();
        ar & boost::serialization::base_object<BaseLayer>(*this);
        ar & this->sizeOfFilterMatrix;
        ar & this->shapeOfInput;
    }
}