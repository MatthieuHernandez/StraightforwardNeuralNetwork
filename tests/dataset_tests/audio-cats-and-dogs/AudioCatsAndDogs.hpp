#pragma once
#include "../TestDataset.hpp"

class AudioCatsAndDogs final : public TestDataset
{
    private:
        void loadData(std::string folderPath) final;

    public:
        const int sizeOfOneData;
        AudioCatsAndDogs(std::string folderPath, int sizeOfOneData);
};
