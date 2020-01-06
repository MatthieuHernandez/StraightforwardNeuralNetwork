#include "LayerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "AllToAll.hpp"

using namespace std;
using namespace snn;
using namespace internal;

LayerModel snn::AllToAll(int numberOfNeurons, activationFunction activation)
{
    LayerModel model
    {
        allToAll,
        activation,
        numberOfNeurons
    };
    return model;
}

LayerModel snn::Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation)
{
    LayerModel model
    {
        recurrent,
        activation,
        numberOfNeurons,
        numberOfRecurrences,
    };
    return model;
}

LayerModel snn::Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3], activationFunction activation)
{
    LayerModel model
    {
        convolution2D,
        activation,
        -1,
        -1,
        numberOfConvolution,
        sizeOfConvolutionMatrix,
        {sizeOfInputs[0], sizeOfInputs[1], sizeOfInputs[2]}
    };
    return model;
}

inline
unique_ptr<Layer> LayerFactory::build(LayerModel model, int numberOfInputs, float* learningRate, float* momentum)
{
    switch (model.type)
    {
        case allToAll:
            return make_unique<AllToAll>(numberOfInputs,
                                        model.numberOfNeurons,
                                        model.activation,
                                        learningRate,
                                        momentum);

        default:
            throw NotImplementedException("Layer");
    }

}

void LayerFactory::build(vector<unique_ptr<Layer>>& layers, int numberOfInputs, vector<LayerModel>& models, float* learningRate, float* momentum)
{
    int currentNumberofInputs = numberOfInputs;
    for(auto&& model : models)
    {
        layers.push_back(build(model, currentNumberofInputs, learningRate, momentum));
        currentNumberofInputs = model.numberOfNeurons;
    }
}
