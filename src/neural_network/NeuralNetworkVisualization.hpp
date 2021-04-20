#pragma once
#include "layer/FilterLayer.hpp"

namespace snn::internal
{
    class NeuralNetworkVisualization
    {
    private:
        static std::vector<float> getWeights(FilterLayer* filterLayer, int filterNumber);

    public:
        static void saveAsBitmap(FilterLayer* filterLayer, std::string filePath);
        static void saveAsBitmap(FilterLayer* filterLayer, std::vector<float> outputs, std::string filePath);
    };
}
