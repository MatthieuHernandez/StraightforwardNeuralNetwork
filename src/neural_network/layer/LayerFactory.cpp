#include "LayerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "AllToAll.hpp"
#include "Recurrence.hpp"
#include "Convolution1D.hpp"
#include "Convolution2D.hpp"

using namespace std;
using namespace snn;
using namespace internal;

LayerModel snn::AllToAll(int numberOfNeurons, activationFunction activation)
{
    LayerModel model
    {
        allToAll,
        activation,
        -1,
        numberOfNeurons
    };
    return model;
}

LayerModel snn::Recurrence(int numberOfNeurons, int numberOfRecurrences, activationFunction activation)
{
    LayerModel model
    {
        recurrence,
        activation,
        -1,
        numberOfNeurons,
        -1,
        numberOfRecurrences,
    };
    return model;
}

LayerModel snn::Convolution(int numberOfConvolution, int sizeOfConvolutionMatrix, activationFunction activation)
{
    //TODO: Calculate the right number of neurones
    LayerModel model
    {
        convolution,
        activation,
        -1,
        -1,
        -1,
        -1,
        numberOfConvolution,
        sizeOfConvolutionMatrix,
        
    };
    return model;
}

inline
int computeNumberOfInputs(vector<int>& shapeOfInput)
{
    int numberOfInputs = 1;
    for (auto size : shapeOfInput)
        numberOfInputs *= size;
    return numberOfInputs;
}

inline
int computeNumberOfNeuronsForConvolution2D(int sizeOfConvolutionMatrix, int numberOfConvolution, vector<int>& sizeOfInputs)
{
    return numberOfConvolution * (sizeOfInputs[0] - (sizeOfConvolutionMatrix - 1)) * (sizeOfInputs[1] - (sizeOfConvolutionMatrix - 1));
}

inline
int computeNumberOfNeuronsForConvolution1D(int sizeOfConvolutionMatrix, int numberOfConvolution, vector<int>& sizeOfInputs)
{
    return numberOfConvolution * (sizeOfInputs[0] - (sizeOfConvolutionMatrix - 1));
}

inline
unique_ptr<Layer> LayerFactory::build(LayerModel& model, vector<int>& shapeOfInput,
                                      StochasticGradientDescent* optimizer)
{
    model.numberOfInputs = computeNumberOfInputs(shapeOfInput);

    if (shapeOfInput.empty())
        throw InvalidArchitectureException("Input of layer has size of 0.");

    if (model.numberOfInputs > 1000000)
        throw InvalidArchitectureException("Layer is too big.");

    switch (model.type)
    {
    case allToAll:
        if (model.numberOfInputs <= 0)
            throw InvalidArchitectureException("Input of layer has size of 0.");

        model.numberOfInputsByNeurons = model.numberOfInputs;
        return make_unique<AllToAll>(model, optimizer);

    case recurrence:
        if (model.numberOfRecurrences < 0)
            throw InvalidArchitectureException("Input of layer has size of 0.");

        model.numberOfInputsByNeurons = model.numberOfInputs * (model.numberOfRecurrences + 1);
        return make_unique<Recurrence>(model, optimizer);

    case convolution:
        if (shapeOfInput.size() == 1)
        {
            shapeOfInput.push_back(1);
        }
        if (shapeOfInput.size() == 2)
        {
            if (model.sizeOfConvolutionMatrix > shapeOfInput[0])
            {
                throw InvalidArchitectureException("Convolution matrix is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForConvolution1D(model.sizeOfConvolutionMatrix, model.numberOfConvolution, model.shapeOfInput);
            model.numberOfInputsByNeurons = model.sizeOfConvolutionMatrix * model.shapeOfInput[1];
            return make_unique<Convolution1D>(model, optimizer);
        }
        if (shapeOfInput.size() == 3)
        {
            if (model.sizeOfConvolutionMatrix > shapeOfInput[0]
                || model.sizeOfConvolutionMatrix > shapeOfInput[1])
            {
                throw InvalidArchitectureException("Convolution matrix is too big.");
            }
            model.shapeOfInput = shapeOfInput;
            model.numberOfNeurons = computeNumberOfNeuronsForConvolution2D(model.sizeOfConvolutionMatrix, model.numberOfConvolution, model.shapeOfInput);
            model.numberOfInputsByNeurons = model.sizeOfConvolutionMatrix * model.sizeOfConvolutionMatrix * model.shapeOfInput[2];
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
}

void LayerFactory::build(vector<unique_ptr<Layer>>& layers, vector<LayerModel>& models,
                         StochasticGradientDescent* optimizer)
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

    auto& currentSizeOfInputs = models[0].shapeOfInput;
    for (int i = 1; i < models.size(); ++i)
    {
        layers.push_back(build(models[i], currentSizeOfInputs, optimizer));
        currentSizeOfInputs = layers.back()->getShapeOfOutput();
    }
}
