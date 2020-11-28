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
}

inline
unique_ptr<BaseLayer> LocallyConnected2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<LocallyConnected2D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

std::vector<int> LocallyConnected2D::getShapeOfOutput() const
{
    const int restX = shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    return {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + restX,
        this->shapeOfInput[1] / this->sizeOfFilterMatrix + restY,
        this->numberOfFilters
    };
}

int LocallyConnected2D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->sizeOfFilterMatrix * this->sizeOfFilterMatrix * this->shapeOfInput[2])
            return 203;
    }
    return this->FilterLayer::isValid();
}

inline
vector<float> LocallyConnected2D::createInputsForNeuron(int neuronNumber, const vector<float>& inputs) const
{
    vector<float> neuronInputs;
    neuronInputs.reserve(this->neurons[neuronNumber].getNumberOfInputs());
    neuronNumber = neuronNumber % this->getNumberOfNeurons()/this->numberOfFilters;
    const int neuronPositionX = neuronNumber * this->sizeOfFilterMatrix % this->shapeOfInput[0];
    const int neuronPositionY = neuronNumber * this->sizeOfFilterMatrix / this->shapeOfInput[0];

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2]) + neuronPositionX * this->shapeOfInput[2];
        const int endIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2])
        + (neuronPositionX + this->sizeOfFilterMatrix) * this->shapeOfInput[2];
        for (int j = beginIndex; j < endIndex; ++j)
        {
            if(j < (int)inputs.size())
                neuronInputs.push_back(inputs[j]);
            else
                neuronInputs.push_back(inputs[0]);
        }
    }
    return neuronInputs;
}

void LocallyConnected2D::insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const
{
    neuronNumber = neuronNumber % this->getNumberOfNeurons()/this->numberOfFilters;
    const int neuronPositionX = neuronNumber * this->sizeOfFilterMatrix % this->shapeOfInput[0] % this->numberOfFilters;
    const int neuronPositionY = neuronNumber * this->sizeOfFilterMatrix / this->shapeOfInput[0] % this->numberOfFilters;

    for (int i = 0; i < this->sizeOfFilterMatrix; ++i)
    {
        const int beginIndex = ((neuronPositionY + i) * this->shapeOfInput[0] * this->shapeOfInput[2]) + neuronPositionX * this->shapeOfInput[2];
        for(int j = 0; j < this->sizeOfFilterMatrix; ++j)
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
