#pragma once
#include <memory>
#include "data/Data.hpp"

class Dataset
{
public:
     std::unique_ptr<snn::Data> data;

     Dataset() = default;
     ~Dataset() = default;

protected:
    virtual void loadData() = 0;
};