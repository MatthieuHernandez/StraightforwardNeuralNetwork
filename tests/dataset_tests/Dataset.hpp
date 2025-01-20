#pragma once
#include <memory>
#include <snn/data/Data.hpp>
#include <string>

class Dataset
{
    public:
        std::unique_ptr<snn::Data> data;
        ~Dataset() = default;

    protected:
        virtual void loadData(std::string folderPath) = 0;
};
