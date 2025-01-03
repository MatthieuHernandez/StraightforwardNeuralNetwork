#include <fstream>
#include "Mnist.hpp"
#include <snn/data/Data.hpp>
#include <snn/tools/ExtendedExpection.hpp>

using namespace std;
using namespace snn;
using namespace internal;

Mnist::Mnist(string folderPath)
{
    this->loadData(folderPath);
}

void Mnist::loadData(string folderPath)
{
    vector2D<float> trainingInputs = readImages(folderPath + "/train-images.idx3-ubyte", 60000);
    vector2D<float> trainingLabels = readLabels(folderPath + "/train-labels.idx1-ubyte", 60000);
    vector2D<float> testingInputs = readImages(folderPath + "/t10k-images.idx3-ubyte", 10000);
    vector2D<float> testingLabels = readLabels(folderPath + "/t10k-labels.idx1-ubyte", 10000);

    this->data = make_unique<Data>(problem::classification, trainingInputs, trainingLabels, testingInputs, testingLabels);
    this->data->normalize(0, 1);
}

vector2D<float> Mnist::readImages(string filePath, int size)
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
            c = (char)file.get();
            if (shift >= 16)
            {
                float value = (float)static_cast<int>(c);
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

vector2D<float> Mnist::readLabels(string filePath, int size)
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
        c = (char)file.get();
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