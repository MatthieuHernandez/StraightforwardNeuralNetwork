#pragma once
#include "../TestDataset.hpp"

class Iris final : public TestDataset
{
    public:
        explicit Iris(const std::string& folderPath);

    private:
        void loadData(const std::string& folderPath) final;
};
