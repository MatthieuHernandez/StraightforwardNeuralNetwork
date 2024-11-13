#pragma once
#include "../Dataset.hpp"
#include <snn/tools/Tools.hpp>

class Cifar10 final : public Dataset
{
public:
    Cifar10(std::string folderPath);

private:
    void loadData(std::string folderPath) override;
    snn::vector2D<float> readImages(std::string filePaths[], size_t size, snn::vector2D<float>& labels) const;
    static void readImages(std::string filePath, snn::vector2D<float>& images, snn::vector2D<float>& labels);
};