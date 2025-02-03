#include "FashionMnist.hpp"

#include <fstream>
#include <snn/data/Dataset.hpp>
#include <snn/tools/ExtendedExpection.hpp>

using namespace snn;

FashionMnist::FashionMnist(std::string folderPath) { this->loadData(folderPath); }

void FashionMnist::loadData(std::string folderPath)
{
    vector2D<float> trainingInputs = readImages(folderPath + "/train-images-idx3-ubyte", 60000);
    vector2D<float> trainingLabels = readLabels(folderPath + "/train-labels-idx1-ubyte", 60000);
    vector2D<float> testingInputs = readImages(folderPath + "/t10k-images-idx3-ubyte", 10000);
    vector2D<float> testingLabels = readLabels(folderPath + "/t10k-labels-idx1-ubyte", 10000);

    this->dataset = std::make_unique<Dataset>(problem::classification, trainingInputs, trainingLabels, testingInputs,
                                              testingLabels);
    this->dataset->normalize(0, 1);
}

auto FashionMnist::readImages(std::string filePath, int size) -> vector2D<float>
{
    std::ifstream file;
    file.open(filePath, std::ios::in | std::ios::binary);
    vector2D<float> images;
    images.reserve(size);
    constexpr int sizeOfData = 28 * 28;

    if (!file.is_open())
    {
        throw FileOpeningFailedException();
    }

    unsigned char c;
    int shift = 0;
    while (!file.eof())
    {
        const std::vector<float> imageTemp;
        images.push_back(imageTemp);
        images.back().reserve(sizeOfData);
        for (int j = 0; !file.eof() && j < sizeOfData;)
        {
            c = static_cast<char>(file.get());
            if (shift >= 16)
            {
                float value = static_cast<float>(static_cast<int>(c));
                images.back().push_back(value);
                j++;
            }
            else
            {
                shift++;
            }
        }
        if (images.back().size() != sizeOfData) images.resize(images.size() - 1);
    }
    file.close();
    return images;
}

auto FashionMnist::readLabels(std::string filePath, int size) -> vector2D<float>
{
    std::ifstream file;
    file.open(filePath, std::ios::in | std::ios::binary);
    vector2D<float> labels;
    labels.reserve(size);

    if (!file.is_open()) throw FileOpeningFailedException();

    unsigned char c;
    int shift = 0;
    while (!file.eof())
    {
        c = static_cast<char>(file.get());
        if (shift >= 8)
        {
            if (!file.eof())
            {
                std::vector<float> labelsTemp(10, 0);
                labels.push_back(labelsTemp);
                labels.back()[c] = 1.0;
            }
        }
        else
        {
            shift++;
        }
    }
    file.close();
    return labels;
}