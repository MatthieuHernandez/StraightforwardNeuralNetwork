#pragma once
#include "../Dataset.hpp"

class AudioCatsAndDogs final : public Dataset
{
private:
    void loadData(std::string folderPath) override;

public:
    AudioCatsAndDogs(std::string folderPath);
};
