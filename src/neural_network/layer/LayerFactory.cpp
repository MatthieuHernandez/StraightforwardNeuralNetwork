#include "LayerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "AllToAll.hpp"

using namespace std;
using namespace snn;
using namespace internal;

LayerModel AllToAll(int numberOfNeurons, activationFunction activation)
{
    LayerModel model
    {
        allToAll,
        numberOfNeurons
    };
    return model;
}

LayerModel Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation)
{
    LayerModel model
    {
        recurrent,
        numberOfNeurons,
        numberOfRecurrences,
    };
    return model;
}

LayerModel Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3], activationFunction activation)
{
    LayerModel model
    {
        convolution2D,
        -1,
        -1,
        numberOfConvolution,
        sizeOfConvolutionMatrix,
    {sizeOfInputs[0], sizeOfInputs[1], sizeOfInputs[2]}
    };
    return model;
}


inline
unique_ptr<Layer> LayerFactory::build(LayerModel model)
{
    switch (model.type)
    {
        case allToAll:
            return make_unique<AllToAll>();

        default:
            throw NotImplementedException("Layer");
    }

}

inline
vector<unique_ptr<Layer>> LayerFactory::build(vector<LayerModel>& models)
{
    vector<unique_ptr<Layer>> layers;
    for(auto&& model : models)
    {
        layers.push_back(build(model));
    }
    return layers;
}

inline
unique_ptr<Layer> LayerFactory::copy(unique_ptr<Layer> layer)
{
    if (typeid(layer) == typeid(AllToAll))
    {
        auto newLayer = make_unique<AllToAll>();
        newLayer->operator=(*layer);
        return newLayer;
    }
    else
    {
        throw NotImplementedException("Layer");
    }
}

inline
vector<unique_ptr<Layer>> LayerFactory::copy(vector<unique_ptr<Layer>> layers)
{
    vector<unique_ptr<Layer>> copiedLayers;
    for (auto&& layer : layers)
    {
        auto copiedLayer = copy(std::move(layer));
        copiedLayers.push_back(copiedLayer);
	}
    return copiedLayers;
}
