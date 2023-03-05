#pragma once
#include <memory>
#include "BaseLayer.hpp"
#include "LayerModel.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "../optimizer/LayerOptimizerModel.hpp"

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
            0,
            {
                0,
                0,
                0,
                0,
                activation::identity
            },
            0,
            0,
            0,
            0,
            {static_cast<int>(sizeOfInput) ...},
            std::vector<LayerOptimizerModel>()
        };
        return model;
    }

    template <class ... TOptimizer>
    LayerModel FullyConnected(int numberOfNeurons, activation activation = activation::sigmoid, TOptimizer ... optimizers)
    {
        LayerModel model
        {
           fullyConnected,
            -1,
            numberOfNeurons,
            -1,
            {
                -1,
                -1,
                -1,
                1.0f,
                activation
            },
            -1,
            -1,
            -1,
            -1,
            std::vector<int>(),
            {static_cast<LayerOptimizerModel>(optimizers) ...}
        };
        return model;
    }

    template <class ... TOptimizer>
    LayerModel Recurrence(int numberOfNeurons, activation activation = activation::tanh, TOptimizer ... optimizers)
    {
        LayerModel model
        {
            recurrence,
            -1,
            numberOfNeurons,
            -1,
            {
                -1,
                -1,
                -1,
                1.0f,
                activation
            },
            -1,
            -1,
            -1,
            -1,
            std::vector<int>(),
            {static_cast<LayerOptimizerModel>(optimizers) ...}
        };
        return model;
    }

    template <class ... TOptimizer>
    LayerModel GruLayer(int numberOfNeurons, TOptimizer ... optimizers)
    {
        LayerModel model
        {
            gruLayer,
            -1,
            numberOfNeurons,
            -1,
            {
                -1,
                -1,
                -1,
                1.0f,
                activation::tanh,
            },
            -1,
            -1,
            -1,
            -1,
            std::vector<int>(),
            {static_cast<LayerOptimizerModel>(optimizers) ...}
        };
        return model;
    }

    template <class ... TOptimizer>
    LayerModel MaxPooling(int kernelSize)
    {
        LayerModel model
        {
            maxPooling,
            -1,
            0,
            -1,
            {
                0,
                0,
                0,
                0.0f,
                activation::identity
            },
            1,
            -1,
            -1,
            kernelSize,
            std::vector<int>(),
            std::vector<LayerOptimizerModel>()

        };
        return model;
    }

    template <class ... TOptimizer>
    LayerModel LocallyConnected(int numberOfLocallyConnected, int kernelSize, activation activation = activation::sigmoid, TOptimizer ... optimizers)
    {
        LayerModel model
        {
            locallyConnected,

            -1,
            -1,
            -1,
            {
                -1,
                -1,
                -1,
                0.001f,
                activation
            },
            numberOfLocallyConnected,
            -1,
            -1,
            kernelSize,
            std::vector<int>(),
            {static_cast<LayerOptimizerModel>(optimizers) ...}

        };
        return model;
    }

    template <class ... TOptimizer>
    LayerModel Convolution(int numberOfConvolution, int kernelSize,
                           activation activation = activation::ReLU, TOptimizer ... optimizers)
    {
        LayerModel model
        {
            convolution,
            -1,
            1,
            -1,
            {
                -1,
                -1,
                -1,
                0.001f,
                activation,
            },
            numberOfConvolution,
            -1,
            -1,
            kernelSize,
            std::vector<int>(),
            {static_cast<LayerOptimizerModel>(optimizers) ...}
        };
        return model;
    }


    namespace internal
    {
        class LayerFactory
        {
        private:
            static std::unique_ptr<BaseLayer> build(LayerModel& model, std::vector<int>& shapeOfInput,
                                                    std::shared_ptr<NeuralNetworkOptimizer> optimizer);

        public:
            static void build(std::vector<std::unique_ptr<BaseLayer>>& layers, std::vector<LayerModel>& models,
                              std::shared_ptr<NeuralNetworkOptimizer> optimizer);
        };
    }
}
