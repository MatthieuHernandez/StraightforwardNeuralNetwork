#pragma once
#include "../TestDataset.hpp"

class DailyMinTemperatures final : public TestDataset
{
    private:
        int numberOfRecurrences;
        void loadData(const std::string& folderPath) final;

    public:
        DailyMinTemperatures(std::string folderPath, int numberOfRecurrences);
};