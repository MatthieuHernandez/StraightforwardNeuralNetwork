#pragma once
#include "../Dataset.hpp"
#include <snn/tools/Tools.hpp>

class FashionMnist final : public Dataset
{
public:
    FashionMnist(std::string folderPath);

private:
    void loadData(std::string folderPath) override;
    static snn::vector2D<float> readImages(std::string filePath, int size);
    static snn::vector2D<float> readLabels(std::string filePath, int size);
};