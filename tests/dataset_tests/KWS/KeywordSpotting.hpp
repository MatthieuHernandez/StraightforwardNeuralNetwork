#pragma once
#include "../TestDataset.hpp"

class KeywordSpotting final : public TestDataset
{
    private:
        void loadData(const std::string& folderPath) final;

    public:
        const int sizeOfOneData;
        KeywordSpotting(const std::string& folderPath, int sizeOfOneData);
};
