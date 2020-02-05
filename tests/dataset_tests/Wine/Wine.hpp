#pragma once
#include "../Dataset.hpp"

class Wine final : public Dataset
{
public:
    Wine(std::string folderPath);

private:
    void loadData(std::string folderPath) override;
};
