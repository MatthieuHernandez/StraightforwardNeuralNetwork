#pragma once
#include <memory>
#include "BaseLayer.hpp"
#include "LayerModel.hpp"
#include "Layer.hpp"
#include "../optimizer/StochasticGradientDescent.hpp"
#include "../optimizer/OptimizerModel.hpp"

namespace snn
{
    template <typename ... TInt>
    extern LayerModel Input(TInt ... sizeOfInput)
    {
        LayerModel model
        {
            input,
            static_cast<int>(activation::sigmoid),
            0,
            {
                0,
                0,
                activation::identity
            },
            0,
            0,
            {static_cast<int>(sizeOfInput) ...},
        };
        return model;
    }

    extern LayerModel FullyConnected(int numberOfNeurons, activation activation = activation::sigmoid);

    extern LayerModel Recurrence(int numberOfNeurons, activation activation = activation::tanh);

    extern LayerModel GruLayer(int numberOfNeurons);

    extern LayerModel LocallyConnected(int numberOfLocallyConnected, int sizeOfLocalMatrix, activation activation = activation::sigmoid);

    extern LayerModel Convolution(int numberOfConvolution, int sizeOfConvolutionMatrix, activation activation = activation::ReLU);

    extern OptimizerModel Dropout(float value);

    namespace internal
    {
        class LayerFactory
        {
        private:
            static std::unique_ptr<BaseLayer> build(LayerModel& model, std::vector<int>& shapeOfInput,
                                                    StochasticGradientDescent* optimizer);

        public:
            static void build(std::vector<std::unique_ptr<BaseLayer>>& layers, std::vector<LayerModel>& models,
                              StochasticGradientDescent* optimizer);
        };
    }
}
