#pragma once
#include "layer/FilterLayer.hpp"

namespace snn::internal
{
    class NeuralNetworkVisualization
    {
    public :
        static void saveAsBitmap(FilterLayer* filterLayer, std::string filePath);

        static char getColorPixel(std::vector<float>& weight, int x, int y, int f);
    };
}
