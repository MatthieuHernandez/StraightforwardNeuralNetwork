#include <boost/serialization/export.hpp>
#include "Convolution2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution2D)

Convolution2D::Convolution2D(LayerModel& model, StochasticGradientDescent* optimizer)
    : Convolution(model, optimizer)
{
}

inline
unique_ptr<Layer> Convolution2D::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<Convolution2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

std::vector<int> Convolution2D::getShapeOfOutput() const
{
    return {
        this->shapeOfInput[0] - (this->sizeOfConvolutionMatrix - 1),
        this->shapeOfInput[1] - (this->sizeOfConvolutionMatrix - 1),
        this->numberOfConvolution
    };
}

int Convolution2D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfConvolutionMatrix * this->sizeOfConvolutionMatrix * this->shapeOfInput[2])
            return 203;
    }
    return this->Convolution::isValid();
}

inline
vector<float> Convolution2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs, bool temporalReset) const
{
    vector<float> neuronInputs;
    neuronInputs.reserve(this->neurons[neuronNumber].getNumberOfInputs());
    neuronNumber = neuronNumber % this->getNumberOfNeurons()/this->numberOfConvolution;
    const int neuronPositionX = neuronNumber % this->shapeOfInput[0];
    const int neuronPositionY = neuronNumber / this->shapeOfInput[0];

    for (int i = 0; i < this->sizeOfConvolutionMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2]) + neuronPositionX * this->shapeOfInput[2];
        const int endIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2])
        + (neuronPositionX + this->sizeOfConvolutionMatrix) * this->shapeOfInput[2];
        for (int j = beginIndex; j < endIndex; ++j)
        {
            neuronInputs.push_back(inputs[j]);
        }
    }
    return neuronInputs;
}

void Convolution2D::insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const
{
    const int neuronPositionX = neuronNumber % this->shapeOfInput[0];
    const int neuronPositionY = neuronNumber / this->shapeOfInput[0];

    for (int i = 0; i < this->sizeOfConvolutionMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[2] * this->shapeOfInput[0]) + neuronPositionX * this->shapeOfInput[2];
        for(int e = 0; e < error.size(); ++e)
        {
            const int index = beginIndex + e;
            errors[index] += error[e];
        }
    }
}

inline
bool Convolution2D::operator==(const Convolution2D& layer) const
{
    return this->Convolution::operator==(layer);
}

inline
bool Convolution2D::operator!=(const Convolution2D& layer) const
{
    return !(*this == layer);
}
