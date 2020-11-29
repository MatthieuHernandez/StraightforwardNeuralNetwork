#pragma once
namespace snn
{
    enum class layerOptimizerType
    {
        dropout = 0,
    };

    enum class neuralNetworkOptimizerType
    {
        stochasticGradientDescent = 0,
        adam = 1
    };
}