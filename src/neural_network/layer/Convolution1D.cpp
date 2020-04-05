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

std::vector<int> Convolution1D::getShapeOfOutput() const
{
    return {
        this->shapeOfInput[0] - (this->sizeOfConvolutionMatrix - 1),
        this->numberOfConvolution
    };
}

int Convolution1D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfConvolutionMatrix * this->shapeOfInput[1])
            return 203;
    }
    return this->Convolution::isValid();
}

inline
vector<float> Convolution1D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    neuronNumber = neuronNumber % this->getNumberOfNeurons()/this->numberOfConvolution;
    const int beginIndex = neuronNumber * this->shapeOfInput[1];
    const int endIndex = (neuronNumber + this->sizeOfConvolutionMatrix) * this->shapeOfInput[1];
    return vector(inputs.begin() + beginIndex, inputs.begin() + endIndex);
}

inline
void Convolution1D::insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error,
                                              std::vector<float>& errors) const
{
    neuronNumber = neuronNumber % this->getNumberOfNeurons()/this->numberOfConvolution;
    const int beginIndex = neuronNumber * this->shapeOfInput[1];
    for (int e = 0; e < error.size(); ++e)
    {
        errors[beginIndex + e] += error[e];
    }
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
