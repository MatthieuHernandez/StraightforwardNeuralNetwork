#include "AudioCatsAndDogs.hpp"

#include <AudioFile.hpp>
#include <array>
#include <snn/tools/ExtendedExpection.hpp>
#include <snn/tools/Tools.hpp>
#include <string>

using namespace snn;

AudioCatsAndDogs::AudioCatsAndDogs(std::string folderPath, int sizeOfOneData)
    : sizeOfOneData(sizeOfOneData)
{
    this->loadData(folderPath);
}

void AudioCatsAndDogs::loadData(const std::string& folderPath)
{
    std::array<std::vector<std::string>, 2> fileNames{
        std::vector<std::string>({
            "/train/cats/cat_1.wav",           "/train/cats/cat_10.wav",          "/train/cats/cat_100.wav",
            "/train/cats/cat_101.wav",         "/train/cats/cat_102.wav",         "/train/cats/cat_103.wav",
            "/train/cats/cat_105.wav",         "/train/cats/cat_106.wav",         "/train/cats/cat_107.wav",
            "/train/cats/cat_108.wav",         "/train/cats/cat_109.wav",         "/train/cats/cat_11.wav",
            "/train/cats/cat_113.wav",         "/train/cats/cat_114.wav",         "/train/cats/cat_116.wav",
            "/train/cats/cat_117.wav",         "/train/cats/cat_118.wav",         "/train/cats/cat_119.wav",
            "/train/cats/cat_12.wav",          "/train/cats/cat_120.wav",         "/train/cats/cat_121.wav",
            "/train/cats/cat_122.wav",         "/train/cats/cat_123.wav",         "/train/cats/cat_124.wav",
            "/train/cats/cat_125.wav",         "/train/cats/cat_127.wav",         "/train/cats/cat_128.wav",
            "/train/cats/cat_13.wav",          "/train/cats/cat_131.wav",         "/train/cats/cat_132.wav",
            "/train/cats/cat_134.wav",         "/train/cats/cat_136.wav",         "/train/cats/cat_138.wav",
            "/train/cats/cat_139.wav",         "/train/cats/cat_140.wav",         "/train/cats/cat_141.wav",
            "/train/cats/cat_142.wav",         "/train/cats/cat_146.wav",         "/train/cats/cat_147.wav",
            "/train/cats/cat_149.wav",         "/train/cats/cat_15.wav",          "/train/cats/cat_150.wav",
            "/train/cats/cat_151.wav",         "/train/cats/cat_153.wav",         "/train/cats/cat_154.wav",
            "/train/cats/cat_155.wav",         "/train/cats/cat_156.wav",         "/train/cats/cat_157.wav",
            "/train/cats/cat_159.wav",         "/train/cats/cat_16.wav",          "/train/cats/cat_160.wav",
            "/train/cats/cat_161.wav",         "/train/cats/cat_162.wav",         "/train/cats/cat_163.wav",
            "/train/cats/cat_164.wav",         "/train/cats/cat_165.wav",         "/train/cats/cat_166.wav",
            "/train/cats/cat_167.wav",         "/train/cats/cat_18.wav",          "/train/cats/cat_19.wav",
            "/train/cats/cat_2.wav",           "/train/cats/cat_21.wav",          "/train/cats/cat_22.wav",
            "/train/cats/cat_23.wav",          "/train/cats/cat_25.wav",          "/train/cats/cat_26.wav",
            "/train/cats/cat_27.wav",          "/train/cats/cat_30.wav",          "/train/cats/cat_31.wav",
            "/train/cats/cat_32.wav",          "/train/cats/cat_33.wav",          "/train/cats/cat_34.wav",
            "/train/cats/cat_35.wav",          "/train/cats/cat_37.wav",          "/train/cats/cat_38.wav",
            "/train/cats/cat_39.wav",          "/train/cats/cat_4.wav",           "/train/cats/cat_40.wav",
            "/train/cats/cat_41.wav",          "/train/cats/cat_43.wav",          "/train/cats/cat_44.wav",
            "/train/cats/cat_45.wav",          "/train/cats/cat_46.wav",          "/train/cats/cat_47.wav",
            "/train/cats/cat_48.wav",          "/train/cats/cat_49.wav",          "/train/cats/cat_5.wav",
            "/train/cats/cat_50.wav",          "/train/cats/cat_51.wav",          "/train/cats/cat_52.wav",
            "/train/cats/cat_53.wav",          "/train/cats/cat_54.wav",          "/train/cats/cat_6.wav",
            "/train/cats/cat_60.wav",          "/train/cats/cat_62.wav",          "/train/cats/cat_63.wav",
            "/train/cats/cat_64.wav",          "/train/cats/cat_65.wav",          "/train/cats/cat_68.wav",
            "/train/cats/cat_69.wav",          "/train/cats/cat_7.wav",           "/train/cats/cat_70.wav",
            "/train/cats/cat_71.wav",          "/train/cats/cat_72.wav",          "/train/cats/cat_73.wav",
            "/train/cats/cat_74.wav",          "/train/cats/cat_77.wav",          "/train/cats/cat_78.wav",
            "/train/cats/cat_8.wav",           "/train/cats/cat_80.wav",          "/train/cats/cat_81.wav",
            "/train/cats/cat_83.wav",          "/train/cats/cat_84.wav",          "/train/cats/cat_87.wav",
            "/train/cats/cat_89.wav",          "/train/cats/cat_9.wav",           "/train/cats/cat_91.wav",
            "/train/cats/cat_92.wav",          "/train/cats/cat_93.wav",          "/train/cats/cat_94.wav",
            "/train/cats/cat_95.wav",          "/train/cats/cat_96.wav",          "/train/cats/cat_97.wav",
            "/train/cats/cat_98.wav",          "/train/cats/cat_99.wav",          "/train/dogs/dog_barking_0.wav",
            "/train/dogs/dog_barking_1.wav",   "/train/dogs/dog_barking_10.wav",  "/train/dogs/dog_barking_100.wav",
            "/train/dogs/dog_barking_101.wav", "/train/dogs/dog_barking_102.wav", "/train/dogs/dog_barking_103.wav",
            "/train/dogs/dog_barking_104.wav", "/train/dogs/dog_barking_105.wav", "/train/dogs/dog_barking_106.wav",
            "/train/dogs/dog_barking_107.wav", "/train/dogs/dog_barking_108.wav", "/train/dogs/dog_barking_109.wav",
            "/train/dogs/dog_barking_11.wav",  "/train/dogs/dog_barking_110.wav", "/train/dogs/dog_barking_111.wav",
            "/train/dogs/dog_barking_13.wav",  "/train/dogs/dog_barking_14.wav",  "/train/dogs/dog_barking_16.wav",
            "/train/dogs/dog_barking_17.wav",  "/train/dogs/dog_barking_18.wav",  "/train/dogs/dog_barking_2.wav",
            "/train/dogs/dog_barking_20.wav",  "/train/dogs/dog_barking_21.wav",  "/train/dogs/dog_barking_22.wav",
            "/train/dogs/dog_barking_23.wav",  "/train/dogs/dog_barking_25.wav",  "/train/dogs/dog_barking_26.wav",
            "/train/dogs/dog_barking_27.wav",  "/train/dogs/dog_barking_28.wav",  "/train/dogs/dog_barking_29.wav",
            "/train/dogs/dog_barking_30.wav",  "/train/dogs/dog_barking_31.wav",  "/train/dogs/dog_barking_32.wav",
            "/train/dogs/dog_barking_33.wav",  "/train/dogs/dog_barking_35.wav",  "/train/dogs/dog_barking_36.wav",
            "/train/dogs/dog_barking_37.wav",  "/train/dogs/dog_barking_38.wav",  "/train/dogs/dog_barking_39.wav",
            "/train/dogs/dog_barking_4.wav",   "/train/dogs/dog_barking_40.wav",  "/train/dogs/dog_barking_41.wav",
            "/train/dogs/dog_barking_42.wav",  "/train/dogs/dog_barking_47.wav",  "/train/dogs/dog_barking_5.wav",
            "/train/dogs/dog_barking_50.wav",  "/train/dogs/dog_barking_51.wav",  "/train/dogs/dog_barking_52.wav",
            "/train/dogs/dog_barking_53.wav",  "/train/dogs/dog_barking_55.wav",  "/train/dogs/dog_barking_56.wav",
            "/train/dogs/dog_barking_57.wav",  "/train/dogs/dog_barking_58.wav",  "/train/dogs/dog_barking_6.wav",
            "/train/dogs/dog_barking_60.wav",  "/train/dogs/dog_barking_61.wav",  "/train/dogs/dog_barking_63.wav",
            "/train/dogs/dog_barking_65.wav",  "/train/dogs/dog_barking_67.wav",  "/train/dogs/dog_barking_68.wav",
            "/train/dogs/dog_barking_69.wav",  "/train/dogs/dog_barking_70.wav",  "/train/dogs/dog_barking_71.wav",
            "/train/dogs/dog_barking_72.wav",  "/train/dogs/dog_barking_74.wav",  "/train/dogs/dog_barking_75.wav",
            "/train/dogs/dog_barking_76.wav",  "/train/dogs/dog_barking_77.wav",  "/train/dogs/dog_barking_79.wav",
            "/train/dogs/dog_barking_80.wav",  "/train/dogs/dog_barking_81.wav",  "/train/dogs/dog_barking_83.wav",
            "/train/dogs/dog_barking_84.wav",  "/train/dogs/dog_barking_85.wav",  "/train/dogs/dog_barking_86.wav",
            "/train/dogs/dog_barking_87.wav",  "/train/dogs/dog_barking_88.wav",  "/train/dogs/dog_barking_92.wav",
            "/train/dogs/dog_barking_93.wav",  "/train/dogs/dog_barking_94.wav",  "/train/dogs/dog_barking_95.wav",
            "/train/dogs/dog_barking_96.wav",  "/train/dogs/dog_barking_97.wav",  "/train/dogs/dog_barking_98.wav",
        }),
        std::vector<std::string>({
            "/test/cats/cat_110.wav",         "/test/cats/cat_112.wav",        "/test/cats/cat_115.wav",
            "/test/cats/cat_126.wav",         "/test/cats/cat_129.wav",        "/test/cats/cat_130.wav",
            "/test/cats/cat_133.wav",         "/test/cats/cat_135.wav",        "/test/cats/cat_137.wav",
            "/test/cats/cat_14.wav",          "/test/cats/cat_143.wav",        "/test/cats/cat_144.wav",
            "/test/cats/cat_148.wav",         "/test/cats/cat_152.wav",        "/test/cats/cat_158.wav",
            "/test/cats/cat_17.wav",          "/test/cats/cat_20.wav",         "/test/cats/cat_24.wav",
            "/test/cats/cat_28.wav",          "/test/cats/cat_29.wav",         "/test/cats/cat_3.wav",
            "/test/cats/cat_36.wav",          "/test/cats/cat_42.wav",         "/test/cats/cat_55.wav",
            "/test/cats/cat_56.wav",          "/test/cats/cat_57.wav",         "/test/cats/cat_58.wav",
            "/test/cats/cat_59.wav",          "/test/cats/cat_61.wav",         "/test/cats/cat_66.wav",
            "/test/cats/cat_67.wav",          "/test/cats/cat_75.wav",         "/test/cats/cat_76.wav",
            "/test/cats/cat_79.wav",          "/test/cats/cat_82.wav",         "/test/cats/cat_85.wav",
            "/test/cats/cat_86.wav",          "/test/cats/cat_88.wav",         "/test/cats/cat_90.wav",
            "/test/dogs/dog_barking_112.wav", "/test/dogs/dog_barking_12.wav", "/test/dogs/dog_barking_15.wav",
            "/test/dogs/dog_barking_19.wav",  "/test/dogs/dog_barking_24.wav", "/test/dogs/dog_barking_3.wav",
            "/test/dogs/dog_barking_34.wav",  "/test/dogs/dog_barking_43.wav", "/test/dogs/dog_barking_44.wav",
            "/test/dogs/dog_barking_45.wav",  "/test/dogs/dog_barking_46.wav", "/test/dogs/dog_barking_48.wav",
            "/test/dogs/dog_barking_49.wav",  "/test/dogs/dog_barking_54.wav", "/test/dogs/dog_barking_59.wav",
            "/test/dogs/dog_barking_62.wav",  "/test/dogs/dog_barking_64.wav", "/test/dogs/dog_barking_66.wav",
            "/test/dogs/dog_barking_7.wav",   "/test/dogs/dog_barking_73.wav", "/test/dogs/dog_barking_78.wav",
            "/test/dogs/dog_barking_8.wav",   "/test/dogs/dog_barking_82.wav", "/test/dogs/dog_barking_89.wav",
            "/test/dogs/dog_barking_9.wav",   "/test/dogs/dog_barking_90.wav", "/test/dogs/dog_barking_91.wav",
            "/test/dogs/dog_barking_99.wav",
        })};
    const int numberOfSet = 2;
    std::array<vector2D<float>, numberOfSet> labels;
    std::array<vector3D<float>, numberOfSet> inputs;

    for (int i = 0; i < numberOfSet; ++i)
    {
        for (const auto& fileName : fileNames.at(i))
        {
            bool isCat{};
            if (fileName.find("cat") != std::string::npos)
            {
                isCat = true;
            }
            else if (fileName.find("dog") != std::string::npos)
            {
                isCat = false;
            }
            else
            {
                throw std::runtime_error("Wrong file name: " + fileName);
            }

            AudioFile<float> audioFile;
            audioFile.load(folderPath + fileName);

            if (audioFile.getNumSamplesPerChannel() == 0)
            {
                throw FileOpeningFailedException();
            }

            const int channel = 0;  // only one
            const int numberOfSamples = audioFile.getNumSamplesPerChannel();

            vector2D<float> dataSound;
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

                    if (isCat)
                    {
                        labels.at(i).push_back({1, 0});
                    }
                    else
                    {
                        labels.at(i).push_back({0, 1});
                    }
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
    this->dataset = std::make_unique<Dataset>(problem::classification, inputs[0], labels[0], inputs[1], labels[1],
                                              nature::sequential);
}
