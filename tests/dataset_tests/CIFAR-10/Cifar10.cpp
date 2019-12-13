#include <fstream>
#include <vector>
#include "Tools.hpp"

using namespace std;
using namespace snn;

void CIFAR_10::loadData()
{
    const string filePaths[6] =
    {
        "./data_batch_1.bin",
        "./data_batch_2.bin",
        "./data_batch_3.bin",
        "./data_batch_4.bin",
        "./data_batch_5.bin"
    };
    vector2D<float> trainingLabels;
    vector2D<float> testingLabels;
    vector2D<float> trainingInputs = this->readImages(filePaths, 5, trainingLabels);
    vector2D<float> testingInputs = this->readImages("./test_batch.bin", 1, testingLabels);
}

vector2D<float> CIFAR_10::readImages(string[] filePaths, int size, vector2D<float>& labels)
{
    vector2D<float> images;
    image.reserve(size * 10000);
    for (int i = 0; i < size; i++)
        this->readImages(path, images, labels);
    return images;
}

void CIFAR_10::readImages(string filePath, vector2D<float>& images, vector2D<float>& labels)
{
    static constexpr sizeOfData = 32*32*3;
    ifstream file;
    file.open(filePaths[i], ios::in | ios::binary);

    if (!file.is_open())
        throw FileOpeningFailed();

    for (int i = 0; !file.eof(); i++)
    {
        char c = file.get();

        const vector<float> labelsTemp(10, 0);
        labels.push_back(labelsTemp);

        if (!file.eof())
            labels.back()[c] = 1.0;
        else
            labels.resize(labels.size() - 1);

        const vector<float> imageTemp;
        images.push_back(imageTemp);
        images.back().reserve(sizeOfData);

        for (int j = 0; !file.eof()  && j < sizeOfData; j++)
        {
            c = file.get();
            const float value = static_cast<int>(c) / 255.0f * 2.0f - 1.0f;
            images.back().push_back(value);
        }
    }
    file.close();
}
