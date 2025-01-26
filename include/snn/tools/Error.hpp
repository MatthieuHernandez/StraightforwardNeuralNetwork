#pragma once
#include <cstdint>

namespace snn
{
enum class errorType : uint8_t
{
    noError = 0,
    neuralNetworkInputTooLarge,
    neuralNetworkTooMuchLayers,
    layerTooMuchNeurons,
    layerWrongNumberOfInputs,
    layerWrongNumberOfOutputs,
    filterLayerWrongNumberOfOutputs,
    conv1DLayerWrongNumberOfInputs,
    conv2DLayerWrongNumberOfInputs,
    locallyConnected1DWrongNumberOfInputs,
    locallyConnected2DWrongNumberOfInputs,
    maxPooling1DWrongNumberOfInputs,
    maxPooling2DWrongNumberOfInputs,
    neuronWrongBias,
    neuronWrongWeight,
    neuronTooMuchWeigths,
    recurrentNeuronWrongNumberOfWeight,
    optimizerWrongLearningRate,
    optimizerWrongMomentum,
    dataSetNull,
    dataWrongLabelSize,
    dataWrongIdexes,
    dataWrongSize,
    temporalCompositeSetNull,
    compositeForNonTemporalDataEmpty,
    compositeForTemporalDataEmpty,
    compositeForTimeSeriesEmpty
};
}  // namespace snn