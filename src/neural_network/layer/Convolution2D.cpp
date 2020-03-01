#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "Convolution2D.hpp"

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
int Convolution2D::computeNumberOfNeurons(int numberOfConvolution, std::array<int, 3> sizeOfInputs)
{
    return numberOfConvolution * sizeOfInputs[0] * sizeOfInputs[1];
}

inline
int Convolution2D::computeNumberOfInputsForNeurones(int sizeOfConvolutionMatrix, std::array<int, 3> sizeOfInputs)
{
    return sizeOfConvolutionMatrix * sizeOfConvolutionMatrix * sizeOfInputs[2];
}

Convolution2D::Convolution2D(int numberOfConvolution,
                             int sizeOfConvolutionMatrix,
                             std::array<int, 3> sizeOfInputs,
                             activationFunction activation,
                             StochasticGradientDescent* optimizer)
     : Layer(convolution2D, computeNumberOfInputs(sizeOfInputs), computeNumberOfNeurons(numberOfConvolution, sizeOfInputs))
{
    this->numberOfConvolution = numberOfConvolution;
    this->sizeOfConvolutionMatrix = sizeOfConvolutionMatrix;
    this->sizeOfInputs = sizeOfInputs;

    for (int n = 0; n < computeNumberOfNeurons(numberOfConvolution, sizeOfInputs); ++n)
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

inline
vector<float> Convolution2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs)
{
    return ;
}
