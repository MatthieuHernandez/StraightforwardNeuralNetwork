#pragma once
#include "../Dataset.hpp"
#include "tools/Tools.hpp"

class FashionMnist final : public Dataset
{
public :
    FashionMnist();

private :
    void loadData() override;
    snn::vector2D<float> readImages(std::string filePath, int size);
    snn::vector2D<float> readLabels(std::string filePath, int size);
};