#include <boost/serialization/export.hpp>
#include "Convolution2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;
using namespace tools;

BOOST_CLASS_EXPORT(Convolution2D)

Convolution2D::Convolution2D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    this->shapeOfOutput = {
        this->shapeOfInput[0] - (this->kernelSize - 1),
        this->shapeOfInput[1] - (this->kernelSize - 1),
        this->numberOfFilters
    };
    this->sizeOfNeuronInputs = this->kernelSize * this->kernelSize * this->shapeOfInput[2];
    this->neuronInputs.resize(this->sizeOfNeuronInputs);
}

inline
unique_ptr<BaseLayer> Convolution2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<Convolution2D>(*this);
    for (auto& neuron : layer->neurons)
        neuron.setOptimizer(optimizer);
    return layer;
}

int Convolution2D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfNeuronInputs)
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> Convolution2D::createInputsForNeuron(const int neuronIndex, const vector<float>& inputs)
{
    const int neuronPosX = neuronIndex % this->shapeOfOutput[0];
    const int neuronPosY = neuronIndex / this->shapeOfOutput[0];

    for (int i = 0; i < this->kernelSize; ++i)
    {
        const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] + neuronPosX) * this->shapeOfInput[2];
        for (int j = 0; j < this->kernelSize; ++j)
        {
            for (int k = 0; k < this->shapeOfInput[2]; ++k)
            {
                const int indexErrors = beginIndex + (j * this->shapeOfInput[2] + k);
                const int indexKernel = (i * this->kernelSize + j) * this->shapeOfInput[2] + k;
                this->neuronInputs[indexKernel] = inputs[indexErrors];
            }
        }
    }
    return this->neuronInputs;
}

inline
void Convolution2D::insertBackOutputForNeuron(const int neuronIndex, const std::vector<float>& error, std::vector<float>& errors)
{
    const int neuronPosX = neuronIndex % this->shapeOfOutput[0];
    const int neuronPosY = neuronIndex / this->shapeOfOutput[0];

    for (int i = 0; i < this->kernelSize; ++i)
    {
        const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] + neuronPosX) * this->shapeOfInput[2];
        for (int j = 0; j < this->kernelSize; ++j)
        {
            for (int k = 0; k < this->shapeOfInput[2]; ++k)
            {
                const int indexErrors = beginIndex + (j * this->shapeOfInput[2] + k);
                const int indexKernel = (i * this->kernelSize + j) * this->shapeOfInput[2] + k;
                errors[indexErrors] += error[indexKernel];
            }
        }
    }
}

inline
bool Convolution2D::operator==(const BaseLayer& layer) const
{
        try
    {
        const auto& f = dynamic_cast<const Convolution2D&>(layer);
        return this->FilterLayer::operator==(layer)
            && this->sizeOfNeuronInputs == f.sizeOfNeuronInputs
            && this->neuronInputs == f.neuronInputs;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

inline
bool Convolution2D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
