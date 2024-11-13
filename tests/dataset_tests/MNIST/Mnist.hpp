#pragma once
#include "../Dataset.hpp"
#include <snn/tools/Tools.hpp>

class Mnist final : public Dataset
{
public:
    Mnist(std::string folderPath);

private:
    void loadData(std::string folderPath) override;
    static snn::vector2D<float> readImages(std::string filePath, int size);
    snn::vector2D<float> readLabels(std::string filePath, int size);
};