#pragma once
#include <random>
#include <vector>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "LayerOptimizer.hpp"

namespace snn::internal
{
    class Dropout final : public LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float value;
        float reverseValue;
        std::uniform_real_distribution<> dist;
        std::vector<bool> presenceProbabilities;

    public:
        Dropout() = default;  // use restricted to Boost library only
        Dropout(float value, BaseLayer* layer);
        Dropout(const Dropout& dropout, const BaseLayer* layer);
        ~Dropout() override = default;

        std::unique_ptr<LayerOptimizer> clone(const BaseLayer* newLayer) const override;

        void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) override;
        void applyAfterOutputForTesting(std::vector<float>& outputs) override;

        void applyBeforeBackpropagation(std::vector<float>& inputErrors) override;

        bool operator==(const LayerOptimizer& optimizer) const override;
        bool operator!=(const LayerOptimizer& optimizer) const override;
    };

    template <class Archive>
    void Dropout::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<Dropout, LayerOptimizer>();
        ar & boost::serialization::base_object<LayerOptimizer>(*this);
        ar & this->value;
        ar & this->reverseValue;
        ar & this->presenceProbabilities;
    }
}
