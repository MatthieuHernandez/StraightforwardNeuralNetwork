#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "../layer/BaseLayer.hpp"

namespace snn::internal
{
class LayerOptimizer
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        const BaseLayer* layer;

    public:
        LayerOptimizer() = default;  // use restricted to Boost library only
        LayerOptimizer(const BaseLayer* layer);
        virtual ~LayerOptimizer() = default;

        virtual auto clone(const BaseLayer* layer) const -> std::unique_ptr<LayerOptimizer> = 0;

        virtual void applyAfterOutputForTraining(std::vector<float>& outputs, bool temporalReset) = 0;
        virtual void applyAfterOutputForTesting(std::vector<float>& outputs) = 0;

        virtual void applyBeforeBackpropagation(std::vector<float>& inputErrors) = 0;

        [[nodiscard]] virtual auto summary() const -> std::string = 0;

        virtual auto operator==(const LayerOptimizer& optimizer) const -> bool;
        virtual auto operator!=(const LayerOptimizer& optimizer) const -> bool;
};

template <class Archive>
void LayerOptimizer::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    ar& this->layer;
}
}  // namespace snn::internal
