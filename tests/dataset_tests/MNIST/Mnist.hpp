#pragma once
#include "../Dataset.hpp"

class Mnist final : public Dataset
{
private :
    void loadData() override;
    vector2D<float> readImages(std::ifstream& Images);
    vector2D<float> readLabels(std::ifstream& labels);
};