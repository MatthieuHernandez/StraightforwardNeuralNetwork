#pragma once
#include "../Dataset.hpp"
#include "tools/Tools.hpp"

class Cifar10 final : public Dataset
{
public :
    Cifar10();

private :
    void loadData() override;
    snn::vector2D<float> readImages(std::string filePaths[], int size, snn::vector2D<float>& labels);
    void readImages(std::string filePath, snn::vector2D<float>& images, snn::vector2D<float>& labels);
};