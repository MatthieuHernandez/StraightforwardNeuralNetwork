#include <string>
#include "AudioCatsAndDogs.hpp"
#include "tools/Tools.hpp"
#include "AudioFile.h"

using namespace std;
using namespace snn;
using namespace internal;

AudioCatsAndDogs::AudioCatsAndDogs(std::string folderPath)
{
    this->loadData(folderPath);
}

void AudioCatsAndDogs::loadData(std::string folderPath)
{
    vector<string> fileNames[2];

    vector2D<float> labels[2];
    vector3D<float> inputs[2];

    for (int i = 0; i < 2; ++i)
    {
        for (auto fileName : fileNames[i])
        {
            AudioFile<float> audioFile;
            audioFile.load(folderPath + fileName); ////////

            const int channel = 0; // only one
            const int numberOfSamples = audioFile.getNumSamplesPerChannel();

            vector2D<float> dataSound;
            dataSound.reserve(numberOfSamples/16);

            for (int i = 0; i < numberOfSamples; i++) // Sample Rate 16000
            {
                if (i % 16 == 0)
                {
                    dataSound.push_back({});
                    dataSound.back().reserve(16);
                }
                const float sample = audioFile.samples[channel][i];
                dataSound.back().push_back(sample);
            }
            inputs[i].push_back(dataSound);

            if (fileName.find("cat") != std::string::npos)
            {
                labels[i].push_back({1, 0});
            }
            if (fileName.find("dog") != std::string::npos)
            {
                labels[i].push_back({0, 1});
            }
            else
            {
                throw runtime_error("Wrong file name: " + fileName);
            }
        }
    }
    this->data = make_unique<Data>(classification,
                                   inputs[0],
                                   labels[0],
                                   inputs[1],
                                   labels[1],
                                   temporal);
}
