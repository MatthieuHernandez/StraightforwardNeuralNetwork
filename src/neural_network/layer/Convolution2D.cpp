#include <boost/serialization/export.hpp>
#include "Convolution2D.hpp"
#include "../../tools/ExtendedExpection.hpp"
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

vector<float> Convolution2D::output(const vector<float>& inputs)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

std::vector<float> Convolution2D::backOutput(std::vector<float>& inputsError)
{
    //TODO: adapt for convolution
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& result = neurons[n].backOutput(inputsError[n]);
        for (int r = 0; r < numberOfInputs; ++r)
            errors[r] += result[r];
    }
    return {}; //errors;
}

void Convolution2D::train(std::vector<float>& inputsError)
{
    throw NotImplementedException();
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
    return this->Convolution::isValid();
}

inline
vector<float> Convolution2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    vector<float> neuronInputs{};

    for(int i = 0; i < this->sizeOfConvolutionMatrix; ++i)
    {
        const int beginIndex = (neuronNumber * this->shapeOfInput[1]) + i * this->shapeOfInput[0];
        const int endIndex = ((neuronNumber + this->sizeOfConvolutionMatrix) * this->shapeOfInput[1]) + i * this->shapeOfInput[0];
        for (int j = beginIndex; j <= endIndex; ++i)
        {
            neuronInputs.push_back(inputs[j]);
        }
    }
    return neuronInputs;
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
