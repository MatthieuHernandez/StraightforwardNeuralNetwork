#include <boost/serialization/export.hpp>
#include "Convolution.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution)

Convolution::Convolution(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer)
{
    this->numberOfConvolution = model.numberOfConvolution;
    this->sizeOfConvolutionMatrix = model.sizeOfConvolutionMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

vector<float> Convolution::output(const vector<float>& inputs, bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = this->createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(neuronInputs);
    }
    return outputs;
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