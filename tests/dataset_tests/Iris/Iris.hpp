#pragma once
#include "../Dataset.hpp"

class Iris final : public Dataset
{
public :
    Iris();

private :
    void loadData() override;
};

