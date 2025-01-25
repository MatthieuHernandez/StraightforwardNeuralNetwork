#pragma once
#include <memory>

#include "../optimizer/LayerOptimizerModel.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "BaseLayer.hpp"
#include "LayerModel.hpp"

namespace snn
{
template <typename... TInt>
extern auto Input(TInt... sizeOfInput) -> LayerModel
{
    LayerModel model{input,
                     static_cast<int>(activation::sigmoid),
                     0,
                     0,
                     {0, 0, 0, 0, activation::identity},
                     0,
                     0,
                     0,
                     0,
                     {static_cast<int>(sizeOfInput)...},
                     std::vector<LayerOptimizerModel>()};
    return model;
}

template <class... TOptimizer>
auto FullyConnected(int numberOfNeurons, activation activation = activation::sigmoid, TOptimizer... optimizers)
    -> LayerModel
{
    LayerModel model{fullyConnected,
                     -1,
                     numberOfNeurons,
                     -1,
                     {-1, -1, -1, 1.0f, activation},
                     -1,
                     -1,
                     -1,
                     -1,
                     std::vector<int>(),
                     {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto Recurrence(int numberOfNeurons, activation activation = activation::tanh, TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{recurrence,
                     -1,
                     numberOfNeurons,
                     -1,
                     {-1, -1, -1, 1.0f, activation},
                     -1,
                     -1,
                     -1,
                     -1,
                     std::vector<int>(),
                     {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto GruLayer(int numberOfNeurons, TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{gruLayer,
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
                     {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto MaxPooling(int kernelSize) -> LayerModel
{
    LayerModel model{maxPooling,
                     -1,
                     0,
                     -1,
                     {0, 0, 0, 0.0f, activation::identity},
                     1,
                     -1,
                     -1,
                     kernelSize,
                     std::vector<int>(),
                     std::vector<LayerOptimizerModel>()};
    return model;
}

template <class... TOptimizer>
auto LocallyConnected(int numberOfLocallyConnected, int kernelSize, activation activation = activation::sigmoid,
                      TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{locallyConnected,
                     -1,
                     -1,
                     -1,
                     {-1, -1, -1, 0.001f, activation},
                     numberOfLocallyConnected,
                     -1,
                     -1,
                     kernelSize,
                     std::vector<int>(),
                     {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto Convolution(int numberOfConvolution, int kernelSize, activation activation = activation::ReLU,
                 TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{convolution,
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
                     {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

namespace internal
{
class LayerFactory
{
    private:
        static auto build(LayerModel& model, std::vector<int>& shapeOfInput,
                          std::shared_ptr<NeuralNetworkOptimizer> optimizer) -> std::unique_ptr<BaseLayer>;

    public:
        static void build(std::vector<std::unique_ptr<BaseLayer>>& layers, std::vector<LayerModel>& models,
                          std::shared_ptr<NeuralNetworkOptimizer> optimizer);
};
}  // namespace internal
}  // namespace snn
