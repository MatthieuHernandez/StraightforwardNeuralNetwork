#pragma once
#include "../TestDataset.hpp"

class Iris final : public TestDataset
{
    public:
        Iris(std::string folderPath);

    private:
        void loadData(std::string folderPath) final;
};
