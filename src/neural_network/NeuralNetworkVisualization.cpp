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
    auto weights = filterLayer->getWeights();
    auto numberOfFilters = filterLayer->getShapeOfOutput()[2];
    float length = sqrt((float)numberOfFilters);
    int filterX = (int)ceil(length);
    int filterY = (int)ceil(length);

    if (numberOfFilters <= filterX * filterY - filterX)
        filterY--;

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
                        int index = flatten(x * shape[0] + i, y * shape[1] + j, filterX * y + x, shape[0], shape[1]);
                        int value =  (int)weights[index]/255;
                        const char color = value > 255 ? 255 : (char)value;
                        image.set_pixel(x * (shape[0] + 1) + i, y * (shape[1] + 1) + j, color, color, color);
                    }
                }
            }
        }
    }
    image.save_image(filePath);
}
