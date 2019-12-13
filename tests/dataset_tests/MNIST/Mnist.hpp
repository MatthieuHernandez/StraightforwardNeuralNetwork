#pragma once
#include "../Dataset.hpp"

class Mnist final : public Dataset
{
private :
    void loadData() override;
    void readImages(std::ifstream& Images);
    void readLabels(std::ifstream& labels);
};