#pragma once
#include "../Dataset.hpp"
#include "tools/Tools.hpp"

class FashionMnist final : public Dataset
{
public:
    FashionMnist(std::string folderPath);

private:
    void loadData(std::string folderPath) override;
    snn::vector2D<float> readImages(std::string filePath, int size);
    snn::vector2D<float> readLabels(std::string filePath, int size);
};