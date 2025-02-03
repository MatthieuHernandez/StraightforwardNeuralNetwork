#pragma once
#include <snn/tools/Tools.hpp>

#include "../TestDataset.hpp"

class Mnist final : public TestDataset
{
    public:
        Mnist(std::string folderPath);

    private:
        void loadData(std::string folderPath) final;
        static auto readImages(std::string filePath, int size) -> snn::vector2D<float>;
        auto readLabels(std::string filePath, int size) -> snn::vector2D<float>;
};