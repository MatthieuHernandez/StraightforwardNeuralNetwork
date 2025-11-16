#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>

namespace addition
{
auto createNonTemporalDataset(int32_t datasetSize, int32_t numberOfValuesToSum, float precision) -> snn::Dataset;

auto createTimeSeriesDataset(int32_t datasetSize, int32_t numberOfValuesToSum, float precision) -> snn::Dataset;

void trainAndTest(snn::StraightforwardNeuralNetwork& neuralNetwork, snn::Dataset& dataset);
}  // namespace addition
