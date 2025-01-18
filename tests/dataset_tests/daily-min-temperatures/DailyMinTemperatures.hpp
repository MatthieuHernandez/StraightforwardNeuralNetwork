#pragma once
#include "../Dataset.hpp"

class DailyMinTemperatures final : public Dataset
{
    private:
        int numberOfRecurrences;
        void loadData(std::string folderPath) override;

    public:
        DailyMinTemperatures(std::string folderPath, int numberOfRecurrences);
};