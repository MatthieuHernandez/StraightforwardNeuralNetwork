#pragma once
#include "../Dataset.hpp"

class Iris final : public Dataset
{
    public:
        Iris(std::string folderPath);

    private:
        void loadData(std::string folderPath) override;
};
