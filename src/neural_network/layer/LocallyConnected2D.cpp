#include <boost/serialization/export.hpp>
#include "LocallyConnected2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(LocallyConnected2D)

LocallyConnected2D::LocallyConnected2D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    const int restX = shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + restX,
        this->shapeOfInput[1] / this->sizeOfFilterMatrix + restY,
        this->numberOfFilters
    };
    this->precomputedX = this->sizeOfFilterMatrix  % this->shapeOfInput[0];
    this->precomputedY = this->sizeOfFilterMatrix / this->shapeOfInput[0];
    this->sizeOfNeuronInputs = this->sizeOfFilterMatrix * this->sizeOfFilterMatrix * this->shapeOfInput[2];
    this->neuronInputs.resize(this->sizeOfNeuronInputs);
}

inline
unique_ptr<BaseLayer> LocallyConnected2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<LocallyConnected2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

int LocallyConnected2D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfNeuronInputs * this->shapeOfInput[2])
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> LocallyConnected2D::createInputsForNeuron(const int neuronIndex, const vector<float>& inputs)
{
    const int neuronPosX = neuronIndex * this->precomputedX;
    const int neuronPosY = neuronIndex * this->precomputedY;

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] + neuronPosX) * this->shapeOfInput[2];
        for (int j = 0; j < this->sizeOfFilterMatrix; ++j)
        {
            const int indexErrors = beginIndex + j;
            const int indexMatrix = i * this->sizeOfFilterMatrix + j;
            if (indexErrors < (int)inputs.size())
                this->neuronInputs[indexMatrix] = inputs[indexErrors];
            else
                neuronInputs.push_back(0);
        }
    }
    return this->neuronInputs;
}

void LocallyConnected2D::insertBackOutputForNeuron(const int neuronIndex, const std::vector<float>& error, std::vector<float>& errors)
{
    const int neuronPosX = neuronIndex * this->precomputedX;
    const int neuronPosY = neuronIndex * this->precomputedY;

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPosY + i) * this->shapeOfInput[0] + neuronPosX) * this->shapeOfInput[2];
        for (int j = 0; j < this->sizeOfFilterMatrix; ++j)
        {
            const int indexErrors = beginIndex + j;
            const int indexMatrix = i * this->sizeOfFilterMatrix + j;
            errors[indexErrors] += error[indexMatrix];
        }
    }
}

inline
bool LocallyConnected2D::operator==(const BaseLayer& layer) const
{
    return this->FilterLayer::operator==(layer);
}

inline
bool LocallyConnected2D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
