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
int computeNumberOfInputs(vector<int>& shapeOfInput)
{
    int numberOfInputs = 1;
    for (auto size : shapeOfInput)
        numberOfInputs *= size;
    return numberOfInputs;
}

inline
int computeNumberOfOutputsForMaxPooling1D(int sizeOfMatrix, vector<int>& shapeOfInput)
{
    const int rest = shapeOfInput[0] % sizeOfMatrix == 0 ? 0 : 1;

    return ((shapeOfInput[0] / sizeOfMatrix) + rest);
}

inline
int computeNumberOfOutputsForMaxPooling2D(int sizeOfMatrix, vector<int>& shapeOfInput)
{
    const int restX = shapeOfInput[0] % sizeOfMatrix == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % sizeOfMatrix == 0 ? 0 : 1;

    return ((shapeOfInput[0] / sizeOfMatrix) + restX) * ((shapeOfInput[1] / sizeOfMatrix) + restY);
}

inline
int computeNumberOfNeuronsForLocallyConnected1D(int numberOfLocallyConnected, int sizeOfLocalMatrix,
                                                vector<int>& shapeOfInput)
{
    const int rest = shapeOfInput[0] % sizeOfLocalMatrix == 0 ? 0 : 1;

    return numberOfLocallyConnected * ((shapeOfInput[0] / sizeOfLocalMatrix) + rest);
}

inline
int computeNumberOfNeuronsForLocallyConnected2D(int numberOfLocallyConnected, int sizeOfLocalMatrix,
                                                vector<int>& shapeOfInput)
{
    const int restX = shapeOfInput[0] % sizeOfLocalMatrix == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % sizeOfLocalMatrix == 0 ? 0 : 1;

    return numberOfLocallyConnected * ((shapeOfInput[0] / sizeOfLocalMatrix) + restX) * ((shapeOfInput[1] /
        sizeOfLocalMatrix) + restY);
}

inline
int computeNumberOfNeuronsForConvolution1D(int numberOfConvolution, int sizeOfConvolutionMatrix,
                                           vector<int>& shapeOfInput)
{
    return numberOfConvolution * (shapeOfInput[0] - (sizeOfConvolutionMatrix - 1));
}

inline
int computeNumberOfNeuronsForConvolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix,
                                           vector<int>& shapeOfInput)
{
    return numberOfConvolution * (shapeOfInput[0] - (sizeOfConvolutionMatrix - 1)) * (shapeOfInput[1] - (
        sizeOfConvolutionMatrix - 1));
}


inline
unique_ptr<BaseLayer> LayerFactory::build(LayerModel& model, vector<int>& shapeOfInput,
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
        model.neuron.numberOfWeights = model.neuron.numberOfInputs;
        model.numberOfOutputs = model.numberOfNeurons;
        return make_unique<FullyConnected>(model, optimizer);

    case recurrence:
        model.neuron.numberOfInputs = model.numberOfInputs;
        model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1;
        model.numberOfOutputs = model.numberOfNeurons;
        return make_unique<Recurrence>(model, optimizer);

    case gruLayer:
        model.neuron.numberOfInputs = model.numberOfInputs;
        model.neuron.numberOfWeights = model.neuron.numberOfInputs + 1;
        model.numberOfOutputs = model.numberOfNeurons;
        return make_unique<GruLayer>(model, optimizer);

    case maxPooling:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput.push_back(1);
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.sizeOfFilerMatrix > shapeOfInput[0])
            {
                throw InvalidArchitectureException("Matrix of max pooling layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfOutputs = computeNumberOfOutputsForMaxPooling1D(model.sizeOfFilerMatrix, model.shapeOfInput);
            return make_unique<MaxPooling1D>(model);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.sizeOfFilerMatrix > shapeOfInput[0]
                || model.sizeOfFilerMatrix > shapeOfInput[1])
            {
                throw InvalidArchitectureException("Matrix of max pooling layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfOutputs = computeNumberOfOutputsForMaxPooling2D(model.sizeOfFilerMatrix, model.shapeOfInput);
            return make_unique<MaxPooling2D>(model);
        }
        if (shapeOfInput.size() > 3)
            throw InvalidArchitectureException("Input with 3 dimensions or higher is not managed.");
        break;

    case locallyConnected:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput.push_back(1);
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.sizeOfFilerMatrix > shapeOfInput[0])
            {
                throw InvalidArchitectureException("Matrix of locally connected layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForLocallyConnected1D(
                model.numberOfFilters, model.sizeOfFilerMatrix, model.shapeOfInput);
            model.neuron.numberOfInputs = model.sizeOfFilerMatrix * model.shapeOfInput[1];
            model.neuron.numberOfWeights = model.neuron.numberOfInputs;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<LocallyConnected1D>(model, optimizer);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.sizeOfFilerMatrix > shapeOfInput[0]
                || model.sizeOfFilerMatrix > shapeOfInput[1])
            {
                throw InvalidArchitectureException("Matrix of locally connected layer is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForLocallyConnected2D(
                model.numberOfFilters, model.sizeOfFilerMatrix, model.shapeOfInput);
            model.neuron.numberOfInputs = model.sizeOfFilerMatrix * model.sizeOfFilerMatrix * model.shapeOfInput[2];
            model.neuron.numberOfWeights = model.neuron.numberOfInputs;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<LocallyConnected2D>(model, optimizer);
        }
        if (shapeOfInput.size() > 3)
            throw InvalidArchitectureException("Input with 3 dimensions or higher is not managed.");
        break;

    case convolution:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput.push_back(1);
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.sizeOfFilerMatrix > shapeOfInput[0])
            {
                throw InvalidArchitectureException("Convolution matrix is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForConvolution1D(
                model.numberOfFilters, model.sizeOfFilerMatrix, model.shapeOfInput);
            model.neuron.numberOfInputs = model.sizeOfFilerMatrix * model.shapeOfInput[1];
            model.neuron.numberOfWeights = model.neuron.numberOfInputs;
            model.numberOfOutputs = model.numberOfNeurons;
            return make_unique<Convolution1D>(model, optimizer);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.sizeOfFilerMatrix > shapeOfInput[0]
                || model.sizeOfFilerMatrix > shapeOfInput[1])
            {
                throw InvalidArchitectureException("Convolution matrix is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForConvolution2D(
                model.numberOfFilters, model.sizeOfFilerMatrix, model.shapeOfInput);
            model.neuron.numberOfInputs = model.sizeOfFilerMatrix * model.sizeOfFilerMatrix * model.shapeOfInput[2];
            model.neuron.numberOfWeights = model.neuron.numberOfInputs;
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

void LayerFactory::build(vector<unique_ptr<BaseLayer>>& layers, vector<LayerModel>& models,
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
