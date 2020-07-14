#pragma once
#include <memory>

#include "BaseLayer.hpp"
#include "LayerModel.hpp"
#include "Layer.hpp"
#include "../Optimizer.hpp"

namespace snn
{
    template <typename ... TInt>
    extern LayerModel Input(TInt ... sizeOfInput)
    {
        LayerModel model
        {
            input,
            sigmoid,
            0,
            0,
            0,
            0,
            0,
            0,
            {static_cast<int>(sizeOfInput) ...},
        };
        return model;
    };

    extern LayerModel FullyConnected(int numberOfNeurons, activationFunction activation = sigmoid);

    extern LayerModel Recurrence(int numberOfNeurons, int numberOfRecurrences, activationFunction activation = sigmoid);

    extern LayerModel LocallyConnected(int numberOfLocallyConnected, int sizeOfLocalMatrix, activationFunction activation = sigmoid);

    extern LayerModel Convolution(int numberOfConvolution, int sizeOfConvolutionMatrix, activationFunction activation = ReLU);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private:
        static std::unique_ptr<BaseLayer> build(LayerModel& model, std::vector<int>& shapeOfInput, StochasticGradientDescent* optimizer);

    public:
        static void build(std::vector<std::unique_ptr<BaseLayer>>& layers, std::vector<LayerModel>& models, StochasticGradientDescent* optimizer);
    };
}