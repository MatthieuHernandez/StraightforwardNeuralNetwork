#include <fstream>
#include "tools/ExtendedExpection.hpp"
#include "FashionMnist.hpp"
#include "data/Data.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

FashionMnist::FashionMnist(string folderPath)
{
    this->loadData(folderPath);
}

void FashionMnist::loadData(string folderPath)
{
    vector2D<float> trainingInputs = readImages(folderPath + "/train-images-idx3-ubyte", 60000);
    vector2D<float> trainingLabels = readLabels(folderPath + "/train-labels-idx1-ubyte", 60000);
    vector2D<float> testingInputs = readImages(folderPath + "/t10k-images-idx3-ubyte", 10000);
    vector2D<float> testingLabels = readLabels(folderPath + "/t10k-labels-idx1-ubyte", 10000);

    this->data = make_unique<Data>(problem::classification, trainingInputs, trainingLabels, testingInputs, testingLabels);
}

vector2D<float> FashionMnist::readImages(string filePath, int size)
{
    ifstream file;
    file.open(filePath, ios::in | ios::binary);
    vector2D<float> images;
    images.reserve(size);
    constexpr int sizeOfData = 28 * 28;

    if (!file.is_open())
        throw FileOpeningFailedException();

    unsigned char c;
    int shift = 0;
    for (int i = 0; !file.eof(); i++)
    {
        const vector<float> imageTemp;
        images.push_back(imageTemp);
        images.back().reserve(sizeOfData);
        for (int j = 0; !file.eof() && j < sizeOfData;)
        {
            c = file.get();
            if (shift >= 16)
            {
                float value = static_cast<int>(c);
                images.back().push_back(value);
                j++;
            }
            else
                shift ++;
        }
        if (images.back().size() != sizeOfData)
            images.resize(images.size() - 1);
    }
    file.close();
    return images;
}

vector2D<float> FashionMnist::readLabels(string filePath, int size)
{
    ifstream file;
    file.open(filePath, ios::in | ios::binary);
    vector2D<float> labels;
    labels.reserve(size);

    if (!file.is_open())
        throw FileOpeningFailedException();

    unsigned char c;
    int shift = 0;
    for (int i = 0; !file.eof(); i++)
    {
        c = file.get();
        if (shift >= 8)
        {
            if (!file.eof())
            {
                vector<float> labelsTemp(10, 0);
                labels.push_back(labelsTemp);
                labels.back()[c] = 1.0;
            }
        }
        else
            shift ++;
    }
    file.close();
    return labels;
}