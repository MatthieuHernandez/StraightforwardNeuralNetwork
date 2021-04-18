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
    auto numberOfFilters = filterLayer->getShapeOfOutput()[2];
    auto filterSize = filterLayer->getSizeOfFilterMatrix();
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
            auto filterNumber = filterX * y + x;
            auto weights = getWeights(filterLayer, filterNumber);
            for (int i = 0; i < shape[0]; ++i)
            {
                for (int j = 0; j < shape[1]; ++j)
                {
                    if (y * filterX + x < numberOfFilters)
                    {
                        int index = flatten(i, j, shape[0]);
                        int value = static_cast<int>(weights[index] * 128.0f / (filterSize * filterSize)) + 127;
                        const char color = value > 255 ? 255 : value < 0 ? 0 : static_cast<unsigned char>(value);
                        image.set_pixel(x * (shape[0] + 1) + i, y * (shape[1] + 1) + j, color, color, color);
                    }
                }
            }
        }
    }
    image.save_image(filePath);
}

std::vector<float> NeuralNetworkVisualization::getWeights(FilterLayer* filterLayer, int filterNumber)
{
    auto inputShape = filterLayer->getShapeOfInput();
    auto outputShape = filterLayer->getShapeOfOutput();
    auto filterSize = filterLayer->getSizeOfFilterMatrix();
    auto neuronByFilter = outputShape[0] * outputShape[1];
    vector<float> weights(filterLayer->getNumberOfInputs(), 0.0f);
    for (int n = 0; n < neuronByFilter; ++n)
    {
        auto neuronWeight = static_cast<SimpleNeuron*>(filterLayer->getNeuron(neuronByFilter * filterNumber + n))->getWeights();
        const int neuronPositionX = roughenX(n, outputShape[0], outputShape[1]);
        const int neuronPositionY = roughenY(n, outputShape[0], outputShape[1]);

        for (int y = 0; y < filterLayer->getSizeOfFilterMatrix(); ++y)
        {
            for (int x = 0; x < filterLayer->getSizeOfFilterMatrix(); ++x)
            {
                const int i = flatten(neuronPositionX + x, neuronPositionY + y, inputShape[0]);
                const int j = flatten(x, y, filterSize);
                weights[i] += neuronWeight[j];
            }
        }
    }
    return weights;
}
