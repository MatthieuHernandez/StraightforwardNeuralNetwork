#pragma once
#include "LayerType.hpp"
#include "../optimizer/LayerOptimizerModel.hpp"
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
        int sizeOfFilerMatrix;
        int numberOfKernelsPerFilter;
        std::vector<int> shapeOfInput;
        std::vector<LayerOptimizerModel> optimizers;
    };
}
