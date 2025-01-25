#pragma once
#include <snn/tools/Tools.hpp>

#include "../Dataset.hpp"

class Cifar10 final : public Dataset
{
    public:
        Cifar10(std::string folderPath);

    private:
        void loadData(std::string folderPath) final;
        auto readImages(std::string filePaths[], size_t size, snn::vector2D<float>& labels) const
            -> snn::vector2D<float>;
        static void readImages(std::string filePath, snn::vector2D<float>& images, snn::vector2D<float>& labels);
};