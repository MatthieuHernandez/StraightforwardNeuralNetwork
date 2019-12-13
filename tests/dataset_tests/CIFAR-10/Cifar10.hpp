#pragma once
#include "../Dataset.h"

class CIFAR_10 final : public Dataset
{
private :
    void loadData() override;
    vector2D<float> readImages(std::string[] filePaths, int size);
    vector2D<float> readImages(std::string filePath, vector2D<float>& labels);
};