#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "Convolution.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution)

Convolution::Convolution(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(convolution, model.numberOfInputs, model.numberOfNeurons)
{
    this->numberOfConvolution = model.numberOfConvolution;
    this->sizeOfConvolutionMatrix = model.sizeOfConvolutionMatrix;
    this->shapeOfInput = model.shapeOfInput;

    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(this->numberOfInputs, model.activation, optimizer);
    }
}

vector<float> Convolution::output(const vector<float>& inputs)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

std::vector<float> Convolution::backOutput(std::vector<float>& inputsError)
{
    //TODO: adapt for convolution
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& result = neurons[n].backOutput(inputsError[n]);
        for (int r = 0; r < numberOfInputs; ++r)
            errors[r] += result[r];
    }
    return {};//errors;
}

void Convolution::train(std::vector<float>& inputsError)
{
    throw NotImplementedException();
}

int Convolution::isValid() const
{
    return this->Layer::isValid();
}

inline 
bool Convolution::operator==(const Convolution& layer) const
{
    return this->Layer::operator==(layer)
    && this->numberOfConvolution == layer.numberOfConvolution
    && this->sizeOfConvolutionMatrix == layer.sizeOfConvolutionMatrix
    && this->shapeOfInput == layer.shapeOfInput;
}

inline 
bool Convolution::operator!=(const Convolution& layer) const
{
    return !(*this == layer);
}