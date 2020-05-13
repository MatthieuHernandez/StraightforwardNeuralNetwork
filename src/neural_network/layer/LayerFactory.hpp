#pragma once
#include <memory>
#include "LayerModel.hpp"
#include "Layer.hpp"
#include "../Optimizer.hpp"

namespace snn
{
    // Waiting C++20 compatibility
    /*template<typename T>
    concept Int = requires(T a)
    {
        a ->std::template convertible_to<int>;
    };*/

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

    extern LayerModel AllToAll(int numberOfNeurons, activationFunction activation = sigmoid);

    extern LayerModel Recurrence(int numberOfNeurons, int numberOfRecurrences, activationFunction activation = sigmoid);

    extern LayerModel Convolution(int numberOfConvolution, int sizeOfConvolutionMatrix, activationFunction activation = ReLU);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private:
        static std::unique_ptr<Layer> build(LayerModel& model, std::vector<int>& shapeOfInput, StochasticGradientDescent* optimizer);

    public:
        static void build(std::vector<std::unique_ptr<Layer>>& layers, std::vector<LayerModel>& models, StochasticGradientDescent* optimizer);
    };
}