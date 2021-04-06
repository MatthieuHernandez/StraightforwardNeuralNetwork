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
        this->shapeOfInput[0] - (this->sizeOfFilterMatrix - 1),
        this->shapeOfInput[1] - (this->sizeOfFilterMatrix - 1),
        this->numberOfFilters
    };
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
        if (neuron.getNumberOfInputs() != this->sizeOfFilterMatrix * this->sizeOfFilterMatrix * this->shapeOfInput[2])
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> Convolution2D::createInputsForNeuron(const int neuronIndex, const vector<float>& inputs) const
{
    vector<float> neuronInputs;

    const int neuronPosX = neuronIndex % this->shapeOfInput[0];
    const int neuronPosY = neuronIndex / this->shapeOfInput[0];

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] * this->shapeOfInput[2]) + neuronPosX * this->shapeOfInput[2];
        const int endIndex = ((neuronPosY + i) * this->shapeOfInput[0] * this->shapeOfInput[2])
            + (neuronPosX + this->sizeOfFilterMatrix) * this->shapeOfInput[2];
        for (int j = beginIndex; j < endIndex; ++j)
        {
            neuronInputs.push_back(inputs[j]);
        }
    }
    return neuronInputs;
}

void Convolution2D::insertBackOutputForNeuron(const int neuronIndex, const std::vector<float>& error, std::vector<float>& errors) const
{
    const int neuronPosX = neuronIndex % this->shapeOfInput[0];
    const int neuronPosY = neuronIndex / this->shapeOfInput[0];

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] * this->shapeOfInput[2]) + neuronPosX * this->shapeOfInput[2];
        for (int j = 0; j < this->sizeOfFilterMatrix; ++j)
        {
            const int indexErrors = beginIndex + j;
            const int indexMatrix = i * this->sizeOfFilterMatrix + j;
            errors[indexErrors] += error[indexMatrix];
        }
    }
}

inline
bool Convolution2D::operator==(const BaseLayer& layer) const
{
    return this->FilterLayer::operator==(layer);
}

inline
bool Convolution2D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
