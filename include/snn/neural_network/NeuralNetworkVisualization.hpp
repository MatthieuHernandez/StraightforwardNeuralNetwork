#pragma once
#include "layer/FilterLayer.hpp"

namespace snn::internal
{
class NeuralNetworkVisualization
{
    private:
        static auto getWeights(FilterLayer* filterLayer, int filterNumber) -> std::vector<float>;

    public:
        static void saveAsBitmap(FilterLayer* filterLayer, const std::string& filePath);
        static void saveAsBitmap(FilterLayer* filterLayer, std::vector<float> outputs, const std::string& filePath);
        static void saveAsBitmap(std::vector<float> inputs, std::vector<int> shapeOfInput, const std::string& filePath);
};
}  // namespace snn::internal
