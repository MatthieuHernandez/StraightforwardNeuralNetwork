#include "Cifar10.hpp"

#include <array>
#include <fstream>
#include <snn/data/Dataset.hpp>
#include <snn/tools/ExtendedExpection.hpp>

Cifar10::Cifar10(const std::string& folderPath) { this->loadData(folderPath); }

void Cifar10::loadData(const std::string& folderPath)
{
    std::array<std::string, 5> const filePaths = {folderPath + "/data_batch_1.bin", folderPath + "/data_batch_2.bin",
                                                  folderPath + "/data_batch_3.bin", folderPath + "/data_batch_4.bin",
                                                  folderPath + "/data_batch_5.bin"};
    std::array<std::string, 1> const testFilePaths = {folderPath + "/test_batch.bin"};
    snn::vector2D<float> trainingLabels;
    snn::vector2D<float> testingLabels;
    snn::vector2D<float> trainingInputs = this->readImages(filePaths, trainingLabels);
    snn::vector2D<float> testingInputs = this->readImages(testFilePaths, testingLabels);

    this->dataset = std::make_unique<snn::Dataset>(snn::problem::classification, trainingInputs, trainingLabels,
                                                   testingInputs, testingLabels);
    this->dataset->normalize(0, 1);
}

void Cifar10::readImages(const std::string& filePath, snn::vector2D<float>& images, snn::vector2D<float>& labels)
{
    static constexpr int sizeOfData = 32 * 32 * 3;
    std::ifstream file;
    file.open(filePath, std::ios::in | std::ios::binary);  // NOLINT(hicpp-signed-bitwise)

    if (!file.is_open())
    {
        throw snn::FileOpeningFailedException();
    }
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
                imageTemp[((j - 1024) * 3) + 1] = c;
            }
            else
            {
                imageTemp[((j - 2048) * 3) + 2] = c;
            }
        }

        images.push_back(imageTemp);
    }
    file.close();
}
