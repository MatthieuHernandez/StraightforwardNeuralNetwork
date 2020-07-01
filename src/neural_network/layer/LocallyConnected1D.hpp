#pragma once
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "../Optimizer.hpp"
#include "Filter.hpp"

namespace snn::internal
{
    class LocallyConnected1D final : public Filter
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        std::vector<float> createInputsForNeuron(int neuronNumber, const std::vector<float>& inputs) const override;
        void insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const override;

    public:
        LocallyConnected1D() = default; // use restricted to Boost library only
        LocallyConnected1D(LayerModel& model, StochasticGradientDescent* optimizer);
        ~LocallyConnected1D() = default;
        LocallyConnected1D(const LocallyConnected1D&) = default;

        std::unique_ptr<Layer> clone(StochasticGradientDescent* optimizer) const override;

        [[nodiscard]] std::vector<int> getShapeOfOutput() const override;
        [[nodiscard]] int isValid() const override;

        bool operator==(const LocallyConnected1D& layer) const;
        bool operator!=(const LocallyConnected1D& layer) const;
    };

    template <class Archive>
    void LocallyConnected1D::serialize(Archive& ar, const unsigned version)
    {
        boost::serialization::void_cast_register<LocallyConnected1D, Filter>();
        ar & boost::serialization::base_object<Filter>(*this);
    }
}