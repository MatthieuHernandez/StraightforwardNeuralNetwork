#pragma once
#include <memory>
#include <snn/data/Dataset.hpp>
#include <string>

class TestDataset
{
    public:
        std::unique_ptr<snn::Dataset> dataset;
        virtual ~TestDataset() = default;

    protected:
        virtual void loadData(const std::string& folderPath) = 0;
};
