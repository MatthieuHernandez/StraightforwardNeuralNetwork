#pragma once
#include <snn/tools/Tools.hpp>

#include "../TestDataset.hpp"

class FashionMnist final : public TestDataset
{
    public:
        explicit FashionMnist(const std::string& folderPath);

    private:
        void loadData(const std::string& folderPath) final;
        static auto readImages(std::string filePath, int size) -> snn::vector2D<float>;
        static auto readLabels(std::string filePath, int size) -> snn::vector2D<float>;
};