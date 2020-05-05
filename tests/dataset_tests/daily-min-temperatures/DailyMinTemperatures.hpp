#pragma once
#include "../Dataset.hpp"

class DailyMinTemperature final : public Dataset
{
private:
    void loadData(std::string folderPath) override;

public:
    DailyMinTemperature(std::string folderPath);
};