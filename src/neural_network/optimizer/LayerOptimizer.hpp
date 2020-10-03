#pragma once
#include <vector>
#include "Optimizer.hpp"

namespace snn::internal
{
    class LayerOptimizer : public Optimizer
    {
    public:
        virtual void apply(std::vector<float>& output) = 0;
        virtual void applyForBackpropagation(std::vector<float>& output) = 0;
    };
}
