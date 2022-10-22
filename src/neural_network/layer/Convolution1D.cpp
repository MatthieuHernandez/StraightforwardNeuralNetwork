#include <boost/serialization/export.hpp>
#include "Convolution1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution1D)

Convolution1D::Convolution1D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    this->shapeOfOutput = {
        this->shapeOfInput[0] - (this->sizeOfFilterMatrix - 1),
        this->numberOfFilters
    };
}

inline
unique_ptr<BaseLayer> Convolution1D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<Convolution1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

int Convolution1D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfFilterMatrix * this->shapeOfInput[1])
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
Tensor Convolution1D::createInputsForNeuron(const int neuronIndex, const Tensor& inputs)
{
    const int beginIndex = neuronIndex * this->shapeOfInput[1];
    const int endIndex = (neuronIndex + this->sizeOfFilterMatrix) * this->shapeOfInput[1];
    return Tensor(inputs.begin() + beginIndex, inputs.begin() + endIndex);
}

inline
void Convolution1D::insertBackOutputForNeuron(const int neuronIndex,
                                              const Tensor& error,
                                              Tensor& errors)
{
    const int beginIndex = neuronIndex * this->shapeOfInput[1];
    for (int e = 0; e < (int)error.size(); ++e)
    {
        const int i = beginIndex + e;
        errors[i] += error[e];
    }
}

bool Convolution1D::operator==(const BaseLayer& layer) const
{
    return this->FilterLayer::operator==(layer);
}

bool Convolution1D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
