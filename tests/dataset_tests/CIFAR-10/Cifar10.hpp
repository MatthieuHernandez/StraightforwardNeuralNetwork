#pragma once
#include "../Dataset.h"
#include "Tools.hpp"

class CIFAR_10 final : public Dataset
{
private :
    void loadData() override;
    vector2D<float> readImages(std::string[] filePaths, int size, vector2D<float>& labels);
    void readImages(std::string filePath, vector2D<float>& images, vector2D<float>& labels);
};