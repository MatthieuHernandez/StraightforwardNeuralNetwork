#include <fstream>
#include "Cifar10.hpp"
#include "data/Data.hpp"
#include "tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

Cifar10::Cifar10(string folderPath)
{
    this->loadData(folderPath);
}

void Cifar10::loadData(string folderPath)
{
    string filePaths[] =
    {
        folderPath + "/data_batch_1.bin",
        folderPath + "/data_batch_2.bin",
        folderPath + "/data_batch_3.bin",
        folderPath + "/data_batch_4.bin",
        folderPath + "/data_batch_5.bin"
    };
    string testFilePaths[] =
    {
        folderPath + "/test_batch.bin"
    };
    vector2D<float> trainingLabels;
    vector2D<float> testingLabels;
    vector2D<float> trainingInputs = readImages(filePaths, 5, trainingLabels);
    vector2D<float> testingInputs = readImages(testFilePaths, 1, testingLabels);

    this->data = make_unique<Data>(classification, trainingInputs, trainingLabels, testingInputs, testingLabels);
}

vector2D<float> Cifar10::readImages(string filePaths[], int size, vector2D<float>& labels)
{
    vector2D<float> images;
    images.reserve(size * 10000);
    for (int i = 0; i < size; i++)
        this->readImages(filePaths[i], images, labels);
    return images;
}

void Cifar10::readImages(string filePath, vector2D<float>& images, vector2D<float>& labels)
{
    static constexpr int sizeOfData = 32*32*3;
    ifstream file;
    file.open(filePath, ios::in | ios::binary);

    if (!file.is_open())
        throw FileOpeningFailedException();

    for (int i = 0; !file.eof(); i++)
    {
        unsigned char c = file.get();

        const vector<float> labelsTemp(10, 0);
        labels.push_back(labelsTemp);

        if (!file.eof())
            labels.back()[c] = 1.0;
        else
        {
            labels.resize(labels.size() - 1);
            break;
        }

        const vector<float> imageTemp;
        images.push_back(imageTemp);
        images.back().reserve(sizeOfData);

        for (int j = 0; !file.eof()  && j < sizeOfData; j++)
        {
            c = file.get();
            const float value = static_cast<float>(static_cast<unsigned int>(c));
            images.back().push_back(value);
        }
    }
    file.close();
}
