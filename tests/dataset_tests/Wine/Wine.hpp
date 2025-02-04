#pragma once
#include "../TestDataset.hpp"

class Wine final : public TestDataset
{
    private:
        void loadData(std::string folderPath) final;

    public:
        Wine(std::string folderPath);
};
