#pragma once
#include <memory>
#include "LayerModel.hpp"
#include "Layer.hpp"

namespace snn
{
    extern internal::LayerModel AllToAll(int numberOfNeurons, activationFunction activation = sigmoid);

    extern internal::LayerModel Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation = sigmoid);

    extern internal::LayerModel Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3], activationFunction activation = ReLU);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private :
        static std::unique_ptr<Layer> build(LayerModel model, float* learningRate, float* momentum);
        static std::unique_ptr<Layer> copy(const std::unique_ptr<Layer>& layer);

    public :
        static void build(std::vector<std::unique_ptr<Layer>>& layers, int numberOfInput, std::vector<LayerModel>& models, float* learningRate, float* momentum);
        static void copy(std::vector<std::unique_ptr<Layer>>& copiedLayers, const std::vector<std::unique_ptr<Layer>>& layersToCopy);
    };
}