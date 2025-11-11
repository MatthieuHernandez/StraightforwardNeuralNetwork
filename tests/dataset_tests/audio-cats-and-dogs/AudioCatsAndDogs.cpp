#include "AudioCatsAndDogs.hpp"

#include <AudioFile.hpp>
#include <array>
#include <snn/tools/ExtendedExpection.hpp>
#include <snn/tools/Tools.hpp>
#include <string>

AudioCatsAndDogs::AudioCatsAndDogs(const std::string& folderPath, int sizeOfOneData)
    : sizeOfOneData(sizeOfOneData)
{
    this->loadData(folderPath);
}

void AudioCatsAndDogs::loadData(const std::string& folderPath)
{
    std::array<std::vector<std::string>, 2> filePaths;
    filePaths[0] = snn::tools::getFilePaths(folderPath + "/train", ".wav");
    filePaths[1] = snn::tools::getFilePaths(folderPath + "/test", ".wav");

    const int numberOfSet = 2;
    std::array<snn::vector2D<float>, numberOfSet> labels;
    std::array<snn::vector3D<float>, numberOfSet> inputs;

    for (int i = 0; i < numberOfSet; ++i)
    {
        for (const auto& filePath : filePaths.at(i))
        {
            std::vector<float> expected;
            if (filePath.find("cat_") != std::string::npos)
            {
                expected = {1, 0};
            }
            else if (filePath.find("dog_") != std::string::npos)
            {
                expected = {0, 1};
            }
            else
            {
                throw std::runtime_error("Wrong file: " + filePath);
            }

            AudioFile<float> audioFile;
            audioFile.load(filePath);

            if (audioFile.getNumSamplesPerChannel() == 0)
            {
                throw snn::FileOpeningFailedException();
            }

            const int channel = 0;  // only one
            const int numberOfSamples = audioFile.getNumSamplesPerChannel();

            snn::vector2D<float> dataSound;
            const auto rest = static_cast<int>((numberOfSamples % this->sizeOfOneData) != 0);
            const auto numberOfData = (numberOfSamples / this->sizeOfOneData) + rest;
            dataSound.reserve(numberOfData);
            labels.at(i).reserve(numberOfData);

            for (int j = 0; j < numberOfSamples; j++)  // Sample Rate 16000
            {
                if (j % this->sizeOfOneData == 0)
                {
                    dataSound.emplace_back();
                    dataSound.back().reserve(this->sizeOfOneData);
                    labels.at(i).push_back(expected);
                }
                const float sample = audioFile.samples[channel][j];
                dataSound.back().push_back(sample);
            }
            while (this->sizeOfOneData > static_cast<int>(dataSound.back().size()))
            {
                dataSound.back().push_back(0.0F);
            }
            inputs.at(i).push_back(dataSound);
        }
    }
    this->dataset = std::make_unique<snn::Dataset>(snn::problem::classification, inputs[0], labels[0], inputs[1],
                                                   labels[1], snn::nature::sequential);
}
