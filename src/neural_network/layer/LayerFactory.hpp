#pragma once
#include <cstddef>
#include <memory>
#include "LayerModel.hpp"
#include "Layer.hpp"
#include "../Optimizer.hpp"

namespace snn
{
    template<typename T>
    concept Int = requires(T a)
    {
        a ->std::template convertible_to<int>;
    };

    template<Int ... TInt>
    extern LayerModel Input(Tint ... TArgs);

    extern LayerModel AllToAll(int numberOfNeurons, activationFunction activation = sigmoid);

    extern LayerModel Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation = sigmoid);

    extern LayerModel Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3], activationFunction activation = ReLU);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private:
        static std::unique_ptr<Layer> build(LayerModel model, int numberOfInputs, StochasticGradientDescent* optimizer);

    public:
        static void build(std::vector<std::unique_ptr<Layer>>& layers, int numberOfInputs, std::vector<LayerModel>& models, StochasticGradientDescent* optimizer);
    };
}