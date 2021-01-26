#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "NoNeuronLayer.hpp"

namespace snn::internal
{
    class MaxPooling2D final : public NoNeuronLayer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int numberOfInputs;
        int sizeOfFilterMatrix;
        std::vector<int> shapeOfInput;

        std::vector<float> computeOutput(const std::vector<float>& inputs, bool temporalReset) override;

    public:
        MaxPooling2D() = default; // use restricted to Boost library only
        MaxPooling2D(LayerModel& model);
        MaxPooling2D(const MaxPooling2D&) = default;
        ~MaxPooling2D() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const;

        [[nodiscard]] std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override;
        [[nodiscard]] std::vector<float> outputForBackpropagation(const std::vector<float>& inputs, bool temporalReset) override;

        [[nodiscard]] int getNumberOfInputs() const override;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void MaxPooling2D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<MaxPooling2D, BaseLayer>();
        ar & boost::serialization::base_object<BaseLayer>(*this);
        ar & this->sizeOfFilterMatrix;
        ar & this->shapeOfInput;
    }
}
