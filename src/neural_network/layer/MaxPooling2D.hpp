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
        int sizeOfFilterMatrix;
        std::vector<int> shapeOfInput;
        std::vector<int> shapeOfOutput;

    public:
        MaxPooling2D() = default; // use restricted to Boost library only
        MaxPooling2D(LayerModel& model);
        MaxPooling2D(const MaxPooling2D&) = default;
        ~MaxPooling2D() = default;
        [[nodiscard]] std::unique_ptr<BaseLayer> clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const;

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
    void MaxPooling2D::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<MaxPooling2D, NoNeuronLayer>();
        ar & boost::serialization::base_object<NoNeuronLayer>(*this);
        ar & this->sizeOfFilterMatrix;
        ar & this->shapeOfInput;
    }
}
