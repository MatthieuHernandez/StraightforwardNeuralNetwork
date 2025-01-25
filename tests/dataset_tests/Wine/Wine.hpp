#pragma once
#include "../Dataset.hpp"

class Wine final : public Dataset
{
    private:
        void loadData(std::string folderPath) final;

    public:
        Wine(std::string folderPath);
};
