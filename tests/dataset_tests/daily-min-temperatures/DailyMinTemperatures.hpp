#pragma once
#include "../Dataset.hpp"

class DailyMinTemperatures final : public Dataset
{
    private:
        int numberOfRecurrences;
        void loadData(std::string folderPath) final;

    public:
        DailyMinTemperatures(std::string folderPath, int numberOfRecurrences);
};