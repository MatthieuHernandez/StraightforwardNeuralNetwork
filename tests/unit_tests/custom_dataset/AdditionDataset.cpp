#include "AdditionDataset.hpp"

#include <algorithm>
#include <snn/tools/Tools.hpp>

#include "../../ExtendedGTest.hpp"

using namespace snn;
namespace addition
{

auto createNonTemporalDataset(int32_t datasetSize, int32_t numberOfValuesToSum, float precision) -> Dataset
{
    vector2D<float> inputData(datasetSize, vector1D<float>());
    vector2D<float> expectedOutputs(datasetSize);

    vector1D<float> values(datasetSize);
    std::ranges::generate(values, []() -> float { return tools::randomBetween(0.0F, 10.0F); });

    for (auto i = 0; i < datasetSize; ++i)
    {
        float sum = 0.0F;
        auto firstIndexSum = i - numberOfValuesToSum + 1;
        for (auto j = firstIndexSum; j <= i; ++j)
        {
            if (j >= 0)
            {
                auto value = values[j];
                inputData[i].push_back(value);
                sum += value;
            }
            else
            {
                inputData[i].push_back(0.0F);
            }
        }
        expectedOutputs[i] = {sum};
    }
    Dataset dataset(problem::regression, inputData, expectedOutputs);
    dataset.setPrecision(precision);
    return dataset;
}

auto createTimeSeriesDataset(int32_t datasetSize, int32_t numberOfValuesToSum, float precision) -> Dataset
{
    vector2D<float> inputData(datasetSize);
    vector2D<float> expectedOutputs(datasetSize);

    vector1D<float> values(datasetSize);
    std::ranges::generate(values, []() -> float { return tools::randomBetween(0.0F, 1.0F); });

    float sum = 0.0F;
    for (auto i = 0; i < datasetSize; ++i)
    {
        auto value = values[i];
        sum += value;
        auto removeIndex = i - numberOfValuesToSum;
        if (removeIndex >= 0)
        {
            sum -= values[removeIndex];
        }

        inputData[i] = {value};

        expectedOutputs[i] = {sum};
    }
    Dataset dataset(problem::regression, inputData, expectedOutputs, nature::timeSeries, numberOfValuesToSum);
    dataset.setPrecision(precision);
    return dataset;
}

void trainAndTest(StraightforwardNeuralNetwork& neuralNetwork, Dataset& dataset)
{
    neuralNetwork.train(dataset, 0.99_acc /*|| 4_s*/);
    auto mae = neuralNetwork.getMeanAbsoluteError();
    auto acc = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(acc, 0.99F);
    ASSERT_MAE(mae, dataset.getPrecision());
}
}  // namespace addition
