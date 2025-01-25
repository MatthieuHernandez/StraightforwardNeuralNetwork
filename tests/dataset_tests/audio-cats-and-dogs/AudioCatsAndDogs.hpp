#pragma once
#include "../Dataset.hpp"

class AudioCatsAndDogs final : public Dataset
{
    private:
        void loadData(std::string folderPath) final;

    public:
        const int sizeOfOneData;
        AudioCatsAndDogs(std::string folderPath, int sizeOfOneData);
};
