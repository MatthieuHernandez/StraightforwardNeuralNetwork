#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "Convolution2D.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution2D)

inline
static int Convolution2D::computeNumberOfInputs(int sizeOfInputs[3])
{
    return sizeOfInputs[0] * sizeOfInputs[1] * sizeOfInputs[2]
}

inline
static int Convolution2D::computeNumberOfNeurons(int numberOfConvolution, int sizeOfInputs[3])
{
    return numberOfConvolution * sizeOfInputs[0] * sizeOfInputs[1];
}

inline
static int Convolution2D::computeNumberOfInputsForNeurones(int sizeOfConvolutionMatrix, int sizeOfInputs[3])
{
    return sizeOfConvolutionMatrix * sizeOfConvolutionMatrix * sizeOfInputs[2];
}

Convolution2D::Convolution2D(int numberOfConvolution,
                             int sizeOfConvolutionMatrix,
                             int sizeOfInputs[3],
                             activationFunction activation,
                             StochasticGradientDescent* optimizer)
     : Layer(Convolution2D, computeNumberOfInputs(sizeOfInputs), computeNumberOfNeurons(numberOfConvolution, sizeOfInputs[3]))
{
    this->numberOfConvolution = numberOfConvolution;
    this->sizeOfConvolutionMatrix = sizeOfConvolutionMatrix;
    this->sizeOfInputs = sizeOfInputs;

    for (int n = 0; n < computeNumberOfNeurons(numberOfConvolution, sizeOfInputs[3]); ++n)
    {
        this->neurons.emplace_back(this->numberOfInputs, this->activation, optimizer);
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
        auto neuronInputs = splitInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

inline
static vector<float> Convolution2D::splitInputsForNeuron(int neuronNumber, const vector<float>& inputs)
{
    
}