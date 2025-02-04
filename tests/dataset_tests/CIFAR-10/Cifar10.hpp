#pragma once
#include <snn/tools/Tools.hpp>

#include "../TestDataset.hpp"

class Cifar10 final : public TestDataset
{
    public:
        explicit Cifar10(std::string folderPath);

    private:
        void loadData(const std::string& folderPath) final;
        template <size_t Size>
        auto readImages(std::array<std::string, Size> filePaths, snn::vector2D<float>& labels) const
            -> snn::vector2D<float>
        {
            snn::vector2D<float> images;
            images.reserve(Size * 10000);
            for (size_t i = 0; i < Size; i++)
            {
                readImages(filePaths[i], images, labels);
            }
            return images;
        }
        static void readImages(const std::string& filePath, snn::vector2D<float>& images, snn::vector2D<float>& labels);
};