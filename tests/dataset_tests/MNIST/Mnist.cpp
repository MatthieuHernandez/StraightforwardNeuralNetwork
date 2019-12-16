#include <fstream>
#include <vector>
#include "Mnist.hpp"
#include "../../../src/tools/Tools.hpp"

using namespace std;
using namespace snn;

void Mnist::loadData()
{
    vector2D<float> trainingInputs = this->readImages("./train-images.idx3-ubyte");
    vector2D<float> trainingLabels = this->readLabes("./train-labels.idx1-ubyte");
    vector2D<float> testingInputs = this->readImages("./t10k-images.idx3-ubyte");
    vector2D<float> testingLabels = this->readLabes("./t10k-labels.idx1-ubyte");

    this->data = new DataForClassification(trainingInputs,
                                           trainingLabels,
                                           testingInputs,
                                           testingLabels)

}

void Mnist::readImages(string filePath, int size)
{
    ifstream file;
    imagesTrainFile.open(filePath, ios::in | ios::binary);
    vector2D<float> images;
    images.reserve(size);

    if (!file.is_open())
        throw FileOpeningFailed();

    unsigned char c;
    int shift = 0;
    for (int i = 0; !file.eof(); i++)
    {
        constexpr vector<float> imageTemp;
        images.push_back(imageTemp);
        images.back().reserve(this->sizeOfData);
        if (!file.eof())
            for (int j = 0; !file.eof() && j < this->sizeOfData;)
            {
                c = file.get();

                if (shift > 15)
                {
                    constexpr float value = static_cast<int>(c) / 255.0f * 2.0f - 1.0f;
                    images.back().push_back(value);
                    j++;
                }
                else
                    shift ++;
            }
    }
    file.close();
    return images;
}

void Mnist::readLabels(string filePath, int size)
{
    ifstream file;
    imagesTrainFile.open(filePath, ios::in | ios::binary);
    vector2D<float>> labels;
    labels.reserve(size);

    if (!file.is_open())
        throw FileOpeningFailed();

    unsigned char c;
    for (int i = 0; !file.eof(); i++)
    {
        c = file.get();

        constexpr vector<float> labelsTemp(10, 0);
        labels.push_back(labelsTemp);

        if (!file.eof())
            labels.back()[c] = 1.0;
        else
            labels.resize(labels.size() - 1);
    }
    file.close();
    return labels;
}