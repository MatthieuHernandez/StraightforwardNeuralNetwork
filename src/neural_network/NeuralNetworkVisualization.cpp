#include "NeuralNetworkVisualization.hpp"

#include <BitmapImage.hpp>

using namespace std;
using namespace snn;
using namespace internal;
using namespace tools;

void NeuralNetworkVisualization::saveAsBitmap(FilterLayer* filterLayer, const string& filePath)
{
    if (filterLayer == nullptr) return;
    auto shape = filterLayer->getShapeOfInput();
    if (shape.size() != 3) return;
    auto numberOfFilters = filterLayer->getShapeOfOutput()[C];
    float length = sqrt((float)numberOfFilters);
    int filterX = (int)ceil(length);
    int filterY = (int)ceil(length);

    if (numberOfFilters <= filterX * filterY - filterX) filterY--;

    bitmap_image image((shape[X] + 1) * filterX - 1, (shape[Y] + 1) * filterY - 1);
    image.set_all_channels(0, 0, 0);
    for (int x = 0; x < filterX; ++x)
    {
        for (int y = 0; y < filterY; ++y)
        {
            if (y * filterX + x < numberOfFilters)
            {
                auto filterNumber = filterX * y + x;
                auto weights = getWeights(filterLayer, filterNumber);
                for (int i = 0; i < shape[X]; ++i)
                {
                    for (int j = 0; j < shape[Y]; ++j)
                    {
                        int index = flatten(i, j, shape[X]);
                        const unsigned char color =
                            static_cast<unsigned char>(round((tanhf(weights[index] / 2.0f) + 1.0f) * 127.5f));
                        image.set_pixel(x * (shape[X] + 1) + i, y * (shape[Y] + 1) + j, color, color, color);
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
    auto filterSize = filterLayer->getKernelSize();
    auto neuronByFilter = outputShape[X] * outputShape[Y];
    vector<float> weights(filterLayer->getNumberOfInputs(), 0.0f);
    for (int n = 0; n < neuronByFilter; ++n)
    {
        auto neuronWeight = static_cast<SimpleNeuron*>(filterLayer->getNeuron(n * (filterNumber + 1)))->getWeights();
        const int neuronPositionX = roughenX(n, outputShape[X], outputShape[Y]);
        const int neuronPositionY = roughenY(n, outputShape[X], outputShape[Y]);

        for (int y = 0; y < filterSize; ++y)
        {
            for (int x = 0; x < filterSize; ++x)
            {
                const int i = flatten(neuronPositionX + x, neuronPositionY + y, inputShape[X]);
                const int j = flatten(x, y, filterSize);
                weights[i] += neuronWeight[j];
            }
        }
    }
    return weights;
}

void NeuralNetworkVisualization::saveAsBitmap(FilterLayer* filterLayer, std::vector<float> outputs,
                                              const string& filePath)
{
    if (filterLayer == nullptr) return;
    auto shape = filterLayer->getShapeOfOutput();
    if (shape.size() != 3) return;
    float length = sqrt((float)shape[C]);
    int filterX = (int)ceil(length);
    int filterY = (int)ceil(length);

    if (shape[C] <= filterX * filterY - filterX) filterY--;

    bitmap_image image((shape[X] + 1) * filterX - 1, (shape[Y] + 1) * filterY - 1);
    image.set_all_channels(0, 0, 0);
    int index = 0;
    for (int j = 0; j < shape[Y]; ++j)
    {
        for (int i = 0; i < shape[X]; ++i)
        {
            for (int y = 0; y < filterY; ++y)
            {
                for (int x = 0; x < filterX; ++x)
                {
                    if (y * filterX + x < shape[C])
                    {
                        const auto color = static_cast<unsigned char>(outputs[index++] * 255.0f);
                        image.set_pixel(x * (shape[X] + 1) + i, y * (shape[Y] + 1) + j, color, color, color);
                    }
                }
            }
        }
    }
    image.save_image(filePath);
}

void NeuralNetworkVisualization::saveAsBitmap(std::vector<float> inputs, std::vector<int> shapeOfInput,
                                              const string& filePath)
{
    if (shapeOfInput.size() != 3) return;
    bitmap_image image(shapeOfInput[X], shapeOfInput[Y]);
    image.set_all_channels(0, 0, 0);
    int index = 0;
    for (int y = 0; y < shapeOfInput[Y]; ++y)
    {
        for (int x = 0; x < shapeOfInput[X]; ++x)
        {
            if (shapeOfInput[C] == 1)
            {
                const auto color = static_cast<unsigned char>(inputs[index++] * 255.0f);
                image.set_pixel(x, y, color, color, color);
            }
            else
            {
                const auto red = static_cast<unsigned char>(inputs[index++] * 255.0f);
                const auto blue = static_cast<unsigned char>(inputs[index++] * 255.0f);
                const auto green = static_cast<unsigned char>(inputs[index++] * 255.0f);
                image.set_pixel(x, y, red, blue, green);
            }
        }
    }
    image.save_image(filePath);
}
