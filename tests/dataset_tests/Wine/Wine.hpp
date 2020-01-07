#pragma once
#include "../Dataset.hpp"

class Wine final : public Dataset
{
public :
    Wine();

private :
    void loadData() override;
};
