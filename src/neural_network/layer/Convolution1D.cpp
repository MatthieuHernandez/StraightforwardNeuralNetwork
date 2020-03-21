#include <boost/serialization/export.hpp>
#include "Convolution1D.hpp"
#include "../../tools/ExtendedExpection.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution1D)

Convolution1D::Convolution1D(LayerModel& model, StochasticGradientDescent* optimizer)
     : Convolution(model, optimizer)
{
}

inline
unique_ptr<Layer> Convolution1D::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<Convolution1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

vector<float> Convolution1D::output(const vector<float>& inputs)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

std::vector<float> Convolution1D::backOutput(std::vector<float>& inputsError)
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

void Convolution1D::train(std::vector<float>& inputsError)
{
    throw NotImplementedException();
}

std::vector<int> Convolution1D::getShapeOfOutput() const
{
    return {
        this->shapeOfInput[0] - (this->sizeOfConvolutionMatrix - 1),
        this->numberOfConvolution
    };
}

int Convolution1D::isValid() const
{
    return this->Layer::isValid();
}

inline
vector<float> Convolution1D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    return {};
}

inline 
bool Convolution1D::operator==(const Convolution1D& layer) const
{
    return this->Convolution::operator==(layer);
}

inline 
bool Convolution1D::operator!=(const Convolution1D& layer) const
{
    return !(*this == layer);
}