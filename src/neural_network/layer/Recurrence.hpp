#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn::internal
{
    //extern template class Layer<Perceptron>;

    class Recurrence final : public Layer<Perceptron>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        std::vector<float> allInputs;
        const int numberOfRecurrences;
        const size_t sizeToCopy;

    protected:
        void addNewInputs(std::vector<float> inputs, bool temporalReset);

    public:
        Recurrence() = default;  // use restricted to Boost library only
        Recurrence(LayerModel& model, StochasticGradientDescent* optimizer);
        Recurrence(const Recurrence&) = default;
        ~Recurrence() = default;
        std::unique_ptr<BaseLayer> clone(StochasticGradientDescent* optimizer) const override;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset) override;
        std::vector<float> backOutput(std::vector<float>& inputErrors) override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const BaseLayer& layer) const override;
        bool operator!=(const BaseLayer& layer) const override;
    };

    template <class Archive>
    void Recurrence::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<Recurrence, Layer>();
        ar & boost::serialization::base_object<Layer>(*this);
    }
}
