#pragma once
#include "../optimizer/LayerOptimizerModel.hpp"
#include "LayerType.hpp"
#include "neuron/NeuronModel.hpp"

namespace snn
{
struct LayerModel
{
        layerType type;
        int numberOfInputs;
        int numberOfNeurons;
        int numberOfOutputs;
        NeuronModel neuron;
        int numberOfFilters;
        int numberOfKernels;
        int numberOfKernelsPerFilter;
        int kernelSize;
        std::vector<int> shapeOfInput;
        std::vector<LayerOptimizerModel> optimizers;
};
}  // namespace snn
