#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "Convolution2D.hpp"


#include <complex.h>
#include <complex.h>

#include "../../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution2D)

inline
int Convolution2D::computeNumberOfInputs(std::array<int, 3> sizeOfInputs)
{
    return sizeOfInputs[0] * sizeOfInputs[1] * sizeOfInputs[2];
}

inline
int Convolution2D::computeNumberOfNeurons(int sizeOfConvolutionMatrix, int numberOfConvolution, std::array<int, 3> sizeOfInputs)
{
    return numberOfConvolution * sizeOfInputs[0] - (sizeOfConvolutionMatrix - 1) * sizeOfInputs[1] - (sizeOfConvolutionMatrix - 1);
}

inline
int Convolution2D::computeNumberOfInputsForNeurones(int sizeOfConvolutionMatrix, std::array<int, 3> sizeOfInputs)
{
    return sizeOfConvolutionMatrix * sizeOfConvolutionMatrix * sizeOfInputs[2];
}

Convolution2D::Convolution2D(int numberOfConvolution,
                             int sizeOfConvolutionMatrix,
                             std::array<int, 3> shapeOfInput,
                             activationFunction activation,
                             StochasticGradientDescent* optimizer)
     : Layer(convolution, computeNumberOfInputs(shapeOfInput), computeNumberOfNeurons(sizeOfConvolutionMatrix, numberOfConvolution, shapeOfInput))
{
    this->numberOfConvolution = numberOfConvolution;
    this->sizeOfConvolutionMatrix = sizeOfConvolutionMatrix;
    this->shapeofInput = shapeOfInput;

    for (int n = 0; n < computeNumberOfNeurons(sizeOfConvolutionMatrix, numberOfConvolution, shapeOfInput); ++n)
    {
        this->neurons.emplace_back(this->numberOfInputs, activation, optimizer);
    }
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
    return {};//errors;
}

void Convolution2D::train(std::vector<float>& inputsError)
{
    throw NotImplementedException();
}

std::vector<int> Convolution2D::getShapeOfOutput() const
{
    return {
        shapeofInput[0] - (this->sizeOfConvolutionMatrix - 1),
        shapeofInput[1] - (this->sizeOfConvolutionMatrix - 1),
        this->numberOfConvolution
    };
}

int Convolution2D::isValid() const
{
    return this->Layer::isValid();
}

inline
vector<float> Convolution2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs)
{
    return {};
}

inline 
bool Convolution2D::operator==(const Convolution2D& layer) const
{
    return this->Layer::operator==(layer)
    && this->numberOfConvolution == layer.numberOfConvolution
    && this->sizeOfConvolutionMatrix == layer.sizeOfConvolutionMatrix
    && this->shapeofInput == layer.shapeofInput;
}

inline 
bool Convolution2D::operator!=(const Convolution2D& layer) const
{
    return !(*this == layer);
}