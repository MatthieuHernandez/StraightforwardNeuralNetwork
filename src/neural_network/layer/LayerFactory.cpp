#include "LayerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "FullyConnected.hpp"
#include "Recurrence.hpp"
#include "GruLayer.hpp"
#include "Convolution1D.hpp"
#include "Convolution2D.hpp"
#include "LocallyConnected1D.hpp"
#include "LocallyConnected2D.hpp"
#include "MaxPooling1D.hpp"
#include "MaxPooling2D.hpp"

using namespace std;
using namespace snn;
using namespace internal;

inline
int computeNumberOfInputs(const vector<int>& shapeOfInput)
{
    int numberOfInputs = 1;
    for (auto size : shapeOfInput)
        numberOfInputs *= size;
    return numberOfInputs;
}

inline
int computeNumberOfOutputsForMaxPooling1D(int kernelSize, const vector<int>& shapeOfInput)
{
    const int rest = shapeOfInput[X] % kernelSize == 0 ? 0 : 1;

    return ((shapeOfInput[X] / kernelSize) + rest);
}

inline
int computeNumberOfOutputsForMaxPooling2D(int kernelSize, const vector<int>& shapeOfInput)
{
    const int restX = shapeOfInput[X] % kernelSize == 0 ? 0 : 1;
    const int restY = shapeOfInput[Y] % kernelSize == 0 ? 0 : 1;

    return shapeOfInput[C] * ((shapeOfInput[X] / kernelSize) + restX) * ((shapeOfInput[Y] / kernelSize) + restY);
}

inline
int computeNumberOfNeuronsForLocallyConnected1D(int numberOfLocallyConnected,
                                                int kernelSize,
                                                const vector<int>& shapeOfInput)
{
    const int rest = shapeOfInput[X] % kernelSize == 0 ? 0 : 1;

    return numberOfLocallyConnected * ((shapeOfInput[X] / kernelSize) + rest);
}

inline
int computeNumberOfNeuronsForLocallyConnected2D(int numberOfLocallyConnected,
                                                int kernelSize,
                                                const vector<int>& shapeOfInput)
{
    const int restX = shapeOfInput[X] % kernelSize == 0 ? 0 : 1;
    const int restY = shapeOfInput[Y] % kernelSize == 0 ? 0 : 1;

    return numberOfLocallyConnected * ((shapeOfInput[X] / kernelSize) + restX) * ((shapeOfInput[Y] /
        kernelSize) + restY);
}

inline
int computeNumberOfKernelsForConvolution1D(int numberOfConvolution,
                                           int kernelSize,
                                           const vector<int>& shapeOfInput)
{
    return numberOfConvolution * (shapeOfInput[X] - (kernelSize - 1));
}

inline
int computeNumberOfKernelsForConvolution2D(int numberOfConvolution,
                                           int kernelSize,
                                           const vector<int>& shapeOfInput)
{
    return numberOfConvolution * (shapeOfInput[X] - (kernelSize - 1)) * (shapeOfInput[Y] - (
        kernelSize - 1));
}


inline
unique_ptr<BaseLayer> LayerFactory::build(LayerModel& model,
                                          vector<int>& shapeOfInput,
                                          shared_ptr<NeuralNetworkOptimizer> optimizer)
{
    model.numberOfInputs = computeNumberOfInputs(shapeOfInput);

    if (shapeOfInput.empty())
        throw InvalidArchitectureException("Input of layer has size of 0.");

    if (model.numberOfInputs > 1000000)
        throw InvalidArchitectureException("Layer is too big.");

    switch (model.type)
    {
    case fullyConnected:
        if (model.numberOfInputs <= 0)
            throw InvalidArchitectureException("Input of layer has size of 0.");

        model.neuron.numberOfInputs = model.numberOfInputs;
        model.neuron.batchSize = 1;
        model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1; // for the bias
        model.numberOfOutputs = model.numberOfNeurons;
        return make_unique<FullyConnected>(model, optimizer);

    case recurrence:
        model.neuron.numberOfInputs = model.numberOfInputs;
        model.neuron.batchSize = 1;
        model.neuron.numberOfWeights = model.neuron.numberOfInputs + 2;
        model.numberOfOutputs = model.numberOfNeurons;
        return make_unique<Recurrence>(model, optimizer);

    case gruLayer:
        model.neuron.numberOfInputs = model.numberOfInputs;
        model.neuron.batchSize = 1;
        model.neuron.numberOfWeights = model.neuron.numberOfInputs + 2;
        model.numberOfOutputs = model.numberOfNeurons;
        return make_unique<GruLayer>(model, optimizer);

    case maxPooling:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput = {1, shapeOfInput[0]};
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.kernelSize > shapeOfInput[X])
            {
                throw InvalidArchitectureException("Kernel of max pooling layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfFilters = shapeOfInput[C];
            model.numberOfOutputs = computeNumberOfOutputsForMaxPooling1D(model.kernelSize, model.shapeOfInput);
            model.numberOfKernels = model.numberOfOutputs;
            return make_unique<MaxPooling1D>(model);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.kernelSize > shapeOfInput[X]
                || model.kernelSize > shapeOfInput[Y])
            {
                throw InvalidArchitectureException("Kernel of max pooling layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfFilters = shapeOfInput[C];
            model.numberOfOutputs = computeNumberOfOutputsForMaxPooling2D(model.kernelSize, model.shapeOfInput);
            model.numberOfKernels = model.numberOfOutputs;
            return make_unique<MaxPooling2D>(model);
        }
        if (shapeOfInput.size() > 3)
            throw InvalidArchitectureException("Input with 3 dimensions or higher is not managed.");
        break;

    case locallyConnected:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput = { 1, shapeOfInput[0] };
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.kernelSize > shapeOfInput[X])
            {
                throw InvalidArchitectureException("Kernel of locally connected layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForLocallyConnected1D(
                model.numberOfFilters,
                model.kernelSize,
                model.shapeOfInput);
            model.numberOfKernels = model.numberOfNeurons;
            model.numberOfKernelsPerFilter = model.numberOfKernels / model.numberOfFilters;
            model.neuron.numberOfInputs = model.kernelSize * model.shapeOfInput[C];
            model.neuron.batchSize = 1;
            model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<LocallyConnected1D>(model, optimizer);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.kernelSize > shapeOfInput[X]
                || model.kernelSize > shapeOfInput[Y])
            {
                throw InvalidArchitectureException("Kernel of locally connected layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForLocallyConnected2D(model.numberOfFilters, model.kernelSize, model.shapeOfInput);
            model.numberOfKernels = model.numberOfNeurons;
            model.numberOfKernelsPerFilter = model.numberOfKernels / model.numberOfFilters;
            model.neuron.numberOfInputs = model.kernelSize * model.kernelSize * model.shapeOfInput[C];
            model.neuron.batchSize = 1;
            model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<LocallyConnected2D>(model, optimizer);
        }
        if (shapeOfInput.size() > 3)
            throw InvalidArchitectureException("Input with 3 dimensions or higher is not managed.");
        break;

    case convolution:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput = { 1, shapeOfInput[0] };
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.kernelSize > shapeOfInput[X])
            {
                throw InvalidArchitectureException("Convolution kernel is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = model.numberOfFilters;
            model.numberOfKernels = computeNumberOfKernelsForConvolution1D(model.numberOfFilters, model.kernelSize, model.shapeOfInput);
            model.numberOfKernelsPerFilter = model.numberOfKernels / model.numberOfFilters;
            model.neuron.numberOfInputs = model.kernelSize * model.shapeOfInput[C];
            model.neuron.batchSize = model.numberOfKernelsPerFilter;
            model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<Convolution1D>(model, optimizer);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.kernelSize > shapeOfInput[X]
                || model.kernelSize > shapeOfInput[Y])
            {
                throw InvalidArchitectureException("Convolution kernel is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = model.numberOfFilters;
            model.numberOfKernels = computeNumberOfKernelsForConvolution2D(model.numberOfFilters, model.kernelSize, model.shapeOfInput);
            model.numberOfKernelsPerFilter = model.numberOfKernels / model.numberOfFilters;
            model.neuron.numberOfInputs = model.kernelSize * model.kernelSize * model.shapeOfInput[C];
            model.neuron.batchSize = model.numberOfKernelsPerFilter;
            model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<Convolution2D>(model, optimizer);
        }
        if (shapeOfInput.size() > 3)
            throw InvalidArchitectureException("Input with 3 dimensions or higher is not managed.");
        break;

    case input:
        throw InvalidArchitectureException("Input LayerModel should be in first position.");

    default:
        throw InvalidArchitectureException("Layer type is not implemented.");
    }
    throw InvalidArchitectureException("The layer factory fail to build layer.");
}

void LayerFactory::build(vector<unique_ptr<BaseLayer>>& layers,
                         vector<LayerModel>& models,
                         shared_ptr<NeuralNetworkOptimizer> optimizer)
{
    if (models.size() > 1000)
        throw InvalidArchitectureException("Too much layers.");

    if (models.empty() || models[0].type != input)
        throw InvalidArchitectureException("First LayerModel must be a Input type LayerModel.");

    if (models.size() < 2)
        throw InvalidArchitectureException("Neural Network must have at least 1 layer.");

    int numberOfInputs = 1;
    for (auto size : models[0].shapeOfInput)
        numberOfInputs *= size;
    if (numberOfInputs > 2073600)
        throw InvalidArchitectureException("Layer is too big.");

    auto& currentShapeOfInput = models[0].shapeOfInput;
    for (size_t i = 1; i < models.size(); ++i)
    {
        layers.push_back(build(models[i], currentShapeOfInput, optimizer));
        currentShapeOfInput = layers.back()->getShapeOfOutput();
    }
}
