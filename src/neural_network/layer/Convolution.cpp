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