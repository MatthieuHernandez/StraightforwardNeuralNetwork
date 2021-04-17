#include <BitmapImage.hpp>
#include "NeuralNetworkVisualization.hpp"

using namespace std;
using namespace snn;
using namespace internal;


void NeuralNetworkVisualization::saveAsBitmap(FilterLayer* filterLayer, string filePath)
{
    if (filterLayer == nullptr)
        return;
    auto shape = filterLayer->getShapeOfInput();
    if (shape.size() != 3)
        return;

    vector<float> weights = {};//filterLayer->getWeights();
    auto numberOfFilters = filterLayer->getShapeOfOutput()[2];
    float length = sqrt((float)numberOfFilters);
    int filterX = (int)floor(length);
    int filterY = (int)floor(length);

    if (length != ceil(sqrt(shape[2])))
        filterX ++;

    bitmap_image image((shape[0] + 1) * filterX - 1, (shape[1] + 1) * filterY - 1);
    image.set_all_channels(0, 0, 0);
    for (int x = 0; x < filterX; ++x)
    {
        for (int y = 0; y < filterY; ++y)
        {
            for (int i = 0; i < shape[0]; ++i)
            {
                for (int j = 0; j < shape[1]; ++j)
                {
                    if (y * filterX + x < numberOfFilters)
                    {
                        const char color = getColorPixel(weights, 0, 0, 0);
                        image.set_pixel(x * (shape[0] + 1) + i, y * (shape[1] + 1) + j, color, color, color);
                    }
                }
            }
        }
    }
    image.save_image(filePath);
}

char NeuralNetworkVisualization::getColorPixel(std::vector<float>& weight, int x, int y, int f)
{
    float w = 0.6f;
    char color = (char)(w * 255.0f);
    return color;
}
