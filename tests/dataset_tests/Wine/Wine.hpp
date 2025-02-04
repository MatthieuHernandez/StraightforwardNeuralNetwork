#pragma once
#include "../TestDataset.hpp"

class Wine final : public TestDataset
{
    private:
        void loadData(const std::string& folderPath) final;

    public:
        explicit Wine(std::string folderPath);
};
