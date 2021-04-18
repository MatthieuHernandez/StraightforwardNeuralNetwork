#pragma once
#include "layer/FilterLayer.hpp"

namespace snn::internal
{
    class NeuralNetworkVisualization
    {
    public :
        static void saveAsBitmap(FilterLayer* filterLayer, std::string filePath);

        static std::vector<float> getWeights(FilterLayer* filterLayer, int filterNumber);
    };
}
