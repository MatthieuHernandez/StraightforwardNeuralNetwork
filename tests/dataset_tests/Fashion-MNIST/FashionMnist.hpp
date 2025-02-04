#pragma once
#include <snn/tools/Tools.hpp>

#include "../TestDataset.hpp"

class FashionMnist final : public TestDataset
{
    public:
        FashionMnist(std::string folderPath);

    private:
        void loadData(std::string folderPath) final;
        static auto readImages(std::string filePath, int size) -> snn::vector2D<float>;
        static auto readLabels(std::string filePath, int size) -> snn::vector2D<float>;
};