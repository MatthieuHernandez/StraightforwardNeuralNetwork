#pragma once
#include "LayerType.hpp"
#include "../optimizer/OptimizerModel.hpp"
#include "neuron/NeuronModel.hpp"

namespace snn
{
    struct LayerModel
    {
        layerType type; 
        int numberOfInputs;
        int numberOfNeurons;
        NeuronModel neuron;
        int numberOfFilters;
        int sizeOfFilerMatrix;
        std::vector<int> shapeOfInput;
        std::vector<OptimizerModel> optimizers;
    };
}
