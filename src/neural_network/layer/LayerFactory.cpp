#include "LayerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "AllToAll.hpp"
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
        numberOfNeurons
    };
    return model;
}

LayerModel snn::Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation)
{
    LayerModel model
    {
        recurrent,
        activation,
        numberOfNeurons,
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
        numberOfConvolution,
        sizeOfConvolutionMatrix
    };
    return model;
}

inline
unique_ptr<Layer> LayerFactory::build(LayerModel& model, vector<int>& shapeOfInput,
                                      StochasticGradientDescent* optimizer)
{
    if (shapeOfInput.empty())
        throw InvalidAchitectureException("Input of layer has size of 0.");

    int numberOfInputs = 1;
    for (auto size : shapeOfInput)
        numberOfInputs *= size;

    if (numberOfInputs > 1000000)
        throw InvalidAchitectureException("Layer is too big.");

    switch (model.type)
    {
    case allToAll:
        if (numberOfInputs <= 0)
            throw InvalidAchitectureException("Input of layer has size of 0.");

        return make_unique<AllToAll>(numberOfInputs,
                                     model.numberOfNeurons,
                                     model.activation,
                                     optimizer);
    case convolution:
        if (shapeOfInput.size() == 1)
            //TODO: to change as convolution1D
            return make_unique<AllToAll>(numberOfInputs,
                                         model.numberOfNeurons,
                                         model.activation,
                                         optimizer);
        if (model.sizeOfConvolutionMatrix > shapeOfInput[0]
        || model.sizeOfConvolutionMatrix > shapeOfInput[1])
        {
            throw InvalidAchitectureException("Convolution matrix is too big.");
        }
        if (shapeOfInput.size() == 3)
        {
            std::array<int, 3> sizeOfInputsParameter = {shapeOfInput[0], shapeOfInput[1], shapeOfInput[2]};
            return make_unique<Convolution2D>(model.numberOfConvolution,
                                              model.sizeOfConvolutionMatrix,
                                              sizeOfInputsParameter,
                                              model.activation,
                                              optimizer);
        }
        if (shapeOfInput.size() > 3)
            throw InvalidAchitectureException("Input with 3 dimensions or higher is not managed.");

    case input:
        throw InvalidAchitectureException("Input LayerModel should be in first position.");

    default:
        throw InvalidAchitectureException("Layer type is not implemented.");
    }
}

void LayerFactory::build(vector<unique_ptr<Layer>>& layers, vector<LayerModel>& models,
                         StochasticGradientDescent* optimizer)
{
    if (models.size() > 1000)
        throw InvalidAchitectureException("Too much layers.");

    if (models.size() < 2)
        throw InvalidAchitectureException("Neural Network must have at least 1 layer.");

    if (models[0].type != input)
        throw InvalidAchitectureException("First LayerModel must be a Input type LayerModel.");

    int numberOfInputs = 1;
    for (auto size : models[0].shapeOfInput)
        numberOfInputs *= size;
    if (numberOfInputs > 2073600)
        throw InvalidAchitectureException("Layer is too big.");

    auto& currentSizeOfInputs = models[0].shapeOfInput;
    for (auto i = 1; i < models.size(); ++i)
    {
        layers.push_back(build(models[i], currentSizeOfInputs, optimizer));
        currentSizeOfInputs = layers.back()->getShapeOfOutput();
    }
}
