#pragma once
#include "../Dataset.hpp"
#include "tools/Tools.hpp"

class Mnist final : public Dataset
{
public :
    Mnist();

private :
    void loadData() override;
    snn::vector2D<float> readImages(std::string filePath, int size);
    snn::vector2D<float> readLabels(std::string filePath, int size);
};