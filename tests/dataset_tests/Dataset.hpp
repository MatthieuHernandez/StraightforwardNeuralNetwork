#pragma once
#include <string>
#include <memory>
#include "data/Data.hpp"

class Dataset
{
public:
     std::unique_ptr<snn::Data> data;
     ~Dataset() = default;

protected:
    virtual void loadData(std::string folderPath) = 0;
};
