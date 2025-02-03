#include "Cifar10.hpp"

#include <array>
#include <fstream>
#include <snn/data/Dataset.hpp>
#include <snn/tools/ExtendedExpection.hpp>

using namespace snn;

Cifar10::Cifar10(std::string folderPath) { this->loadData(folderPath); }

void Cifar10::loadData(const std::string& folderPath)
{
    std::array<std::string, 5> filePaths = {folderPath + "/data_batch_1.bin", folderPath + "/data_batch_2.bin",
                                            folderPath + "/data_batch_3.bin", folderPath + "/data_batch_4.bin",
                                            folderPath + "/data_batch_5.bin"};
    std::array<std::string, 1> testFilePaths = {folderPath + "/test_batch.bin"};
    vector2D<float> trainingLabels;
    vector2D<float> testingLabels;
    vector2D<float> trainingInputs = this->readImages(filePaths, trainingLabels);
    vector2D<float> testingInputs = this->readImages(testFilePaths, testingLabels);

    this->dataset = std::make_unique<Dataset>(problem::classification, trainingInputs, trainingLabels, testingInputs,
                                              testingLabels);
    this->dataset->normalize(0, 1);
}

void Cifar10::readImages(const std::string& filePath, vector2D<float>& images, vector2D<float>& labels)
{
    static constexpr int sizeOfData = 32 * 32 * 3;
    std::ifstream file;
    file.open(filePath, std::ios::in | std::ios::binary);

    if (!file.is_open()) throw FileOpeningFailedException();

    while (!file.eof())
    {
        unsigned char c = static_cast<char>(file.get());

        const std::vector<float> labelsTemp(10, 0);
        labels.push_back(labelsTemp);

        if (!file.eof())
        {
            labels.back()[c] = 1.0;
        }
        else
        {
            labels.resize(labels.size() - 1);
            break;
        }

        std::vector<float> imageTemp;
        imageTemp.resize(sizeOfData, 0);
        for (int j = 0; !file.eof() && j < sizeOfData; j++)
        {
            c = static_cast<char>(file.get());
            // imageTemp[j] = c;
            if (j < 1024)
            {
                imageTemp[j * 3] = c;
            }
            else if (j < 2048)
            {
                imageTemp[(j - 1024) * 3 + 1] = c;
            }
            else
            {
                imageTemp[(j - 2048) * 3 + 2] = c;
            }
        }

        images.push_back(imageTemp);
    }
    file.close();
}
