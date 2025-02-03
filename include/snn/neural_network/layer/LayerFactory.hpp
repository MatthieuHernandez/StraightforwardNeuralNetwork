#pragma once
#include <memory>

#include "../optimizer/LayerOptimizerModel.hpp"
#include "../optimizer/NeuralNetworkOptimizer.hpp"
#include "BaseLayer.hpp"
#include "LayerModel.hpp"

namespace snn
{
const auto almostZero = 0.001F;

template <typename... TInt>
extern auto Input(TInt... sizeOfInput) -> LayerModel
{
    LayerModel model{.type = input,
                     .numberOfInputs = static_cast<int>(activation::sigmoid),
                     .numberOfNeurons = 0,
                     .numberOfOutputs = 0,
                     .neuron = {.numberOfInputs = 0,
                                .batchSize = 0,
                                .numberOfWeights = 0,
                                .bias = 0,
                                .activationFunction = activation::identity},
                     .numberOfFilters = 0,
                     .numberOfKernels = 0,
                     .numberOfKernelsPerFilter = 0,
                     .kernelSize = 0,
                     .shapeOfInput = {static_cast<int>(sizeOfInput)...},
                     .optimizers = {}};
    return model;
}

template <class... TOptimizer>
auto FullyConnected(int numberOfNeurons, activation activation = activation::sigmoid, TOptimizer... optimizers)
    -> LayerModel
{
    LayerModel model{.type = fullyConnected,
                     .numberOfInputs = -1,
                     .numberOfNeurons = numberOfNeurons,
                     .numberOfOutputs = -1,
                     .neuron = {.numberOfInputs = -1,
                                .batchSize = -1,
                                .numberOfWeights = -1,
                                .bias = 1.0F,
                                .activationFunction = activation},
                     .numberOfFilters = -1,
                     .numberOfKernels = -1,
                     .numberOfKernelsPerFilter = -1,
                     .kernelSize = -1,
                     .shapeOfInput = std::vector<int>(),
                     .optimizers = {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto Recurrence(int numberOfNeurons, activation activation = activation::tanh, TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{.type = recurrence,
                     .numberOfInputs = -1,
                     .numberOfNeurons = numberOfNeurons,
                     .numberOfOutputs = -1,
                     .neuron = {.numberOfInputs = -1,
                                .batchSize = -1,
                                .numberOfWeights = -1,
                                .bias = 1.0F,
                                .activationFunction = activation},
                     .numberOfFilters = -1,
                     .numberOfKernels = -1,
                     .numberOfKernelsPerFilter = -1,
                     .kernelSize = -1,
                     .shapeOfInput = std::vector<int>(),
                     .optimizers = {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto GruLayer(int numberOfNeurons, TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{.type = gruLayer,
                     .numberOfInputs = -1,
                     .numberOfNeurons = numberOfNeurons,
                     .numberOfOutputs = -1,
                     .neuron =
                         {
                             .numberOfInputs = -1,
                             .batchSize = -1,
                             .numberOfWeights = -1,
                             .bias = 1.0F,
                             .activationFunction = activation::tanh,
                         },
                     .numberOfFilters = -1,
                     .numberOfKernels = -1,
                     .numberOfKernelsPerFilter = -1,
                     .kernelSize = -1,
                     .shapeOfInput = std::vector<int>(),
                     .optimizers = {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto MaxPooling(int kernelSize) -> LayerModel
{
    LayerModel model{.type = maxPooling,
                     .numberOfInputs = -1,
                     .numberOfNeurons = 0,
                     .numberOfOutputs = -1,
                     .neuron = {.numberOfInputs = 0,
                                .batchSize = 0,
                                .numberOfWeights = 0,
                                .bias = 0.0F,
                                .activationFunction = activation::identity},
                     .numberOfFilters = 1,
                     .numberOfKernels = -1,
                     .numberOfKernelsPerFilter = -1,
                     .kernelSize = kernelSize,
                     .shapeOfInput = std::vector<int>(),
                     .optimizers = std::vector<LayerOptimizerModel>()};
    return model;
}

template <class... TOptimizer>
auto LocallyConnected(int numberOfLocallyConnected, int kernelSize, activation activation = activation::sigmoid,
                      TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{.type = locallyConnected,
                     .numberOfInputs = -1,
                     .numberOfNeurons = -1,
                     .numberOfOutputs = -1,
                     .neuron = {.numberOfInputs = -1,
                                .batchSize = -1,
                                .numberOfWeights = -1,
                                .bias = almostZero,
                                .activationFunction = activation},
                     .numberOfFilters = numberOfLocallyConnected,
                     .numberOfKernels = -1,
                     .numberOfKernelsPerFilter = -1,
                     .kernelSize = kernelSize,
                     .shapeOfInput = std::vector<int>(),
                     .optimizers = {static_cast<LayerOptimizerModel>(optimizers)...}};
    return model;
}

template <class... TOptimizer>
auto Convolution(int numberOfConvolution, int kernelSize, activation activation = activation::ReLU,
                 TOptimizer... optimizers) -> LayerModel
{
    LayerModel model{.type = convolution,
                     .numberOfInputs = -1,
                     .numberOfNeurons = 1,
                     .numberOfOutputs = -1,
                     .neuron =
                         {
                             .numberOfInputs = -1,
                             .batchSize = -1,
                             .numberOfWeights = -1,
                             .bias = almostZero,
                             .activationFunction = activation,
                         },
                     .numberOfFilters = numberOfConvolution,
                     .numberOfKernels = -1,
                     .numberOfKernelsPerFilter = -1,
                     .kernelSize = kernelSize,
                     .shapeOfInput = std::vector<int>(),
                     .optimizers = {static_cast<LayerOptimizerModel>(optimizers)...}};
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
