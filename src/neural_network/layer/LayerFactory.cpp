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
        -1,
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
        -1,
        numberOfConvolution,
        sizeOfConvolutionMatrix,
        {sizeOfInputs[0], sizeOfInputs[1], sizeOfInputs[2]}
    };
    return model;
}

inline
unique_ptr<Layer> LayerFactory::build(LayerModel model, float* learningRate, float* momentum)
{
    switch (model.type)
    {
        case allToAll:
            return make_unique<AllToAll>(model.numberOfInputs,
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
        layers.push_back(build(model, learningRate, momentum));
        currentNumberofInputs = model.numberOfNeurons;
    }
}

inline
unique_ptr<Layer> LayerFactory::copy(const unique_ptr<Layer>& layer)
{
    if (typeid(layer) == typeid(AllToAll))
    {
        auto newLayer = make_unique<AllToAll>();
        newLayer->operator=(*layer);
        return move(newLayer);
    }
    else
    {
        throw NotImplementedException("Layer");
    }
}

void LayerFactory::copy(vector<unique_ptr<Layer>>& copiedLayers, const vector<unique_ptr<Layer>>& layersToCopy)
{
    for (auto& layer : layersToCopy)
    {
        copiedLayers.push_back(copy(layer));
    }
}
