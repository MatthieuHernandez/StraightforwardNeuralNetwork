#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "NoNeuronLayer.hpp"

namespace snn::internal
{
    class MaxPooling1D final : public NoNeuronLayer
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
        MaxPooling1D(const MaxPooling1D&) = default;
        ~MaxPooling1D() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const override;

        [[nodiscard]] Tensor output(const Tensor& inputs, bool temporalReset) override;
        [[nodiscard]] Tensor outputForTraining(const Tensor& inputs, bool temporalReset) override;
        [[nodiscard]] Tensor backOutput(Tensor& inputErrors) override;
        void train(Tensor& inputErrors) override;

        [[nodiscard]] int getNumberOfInputs() const override;
        [[nodiscard]] std::vector<int> getShapeOfInput() const override;
        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;

    };

    template <class Archive>
    void MaxPooling1D::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<MaxPooling1D, NoNeuronLayer>();
        ar & boost::serialization::base_object<NoNeuronLayer>(*this);
        ar & this->sizeOfFilterMatrix;
        ar & this->shapeOfInput;
    }
}