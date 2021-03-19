#include <boost/serialization/export.hpp>
#include "Convolution2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution2D)

Convolution2D::Convolution2D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
}

inline
unique_ptr<BaseLayer> Convolution2D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<Convolution2D>(*this);
    for (auto& neuron : layer->neurons)
        neuron.optimizer = optimizer;
    return layer;
}

std::vector<int> Convolution2D::getShapeOfOutput() const
{
    return {
        this->shapeOfInput[0] - (this->sizeOfFilterMatrix - 1),
        this->shapeOfInput[1] - (this->sizeOfFilterMatrix - 1),
        this->numberOfFilters
    };
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
vector<float> Convolution2D::createInputsForNeuron(const int neuronNumber, const vector<float>& inputs) const
{
    vector<float> neuronInputs;
    neuronInputs.reserve(this->neurons[neuronNumber].getNumberOfInputs());

    const int n = neuronNumber % this->getNumberOfNeurons() / this->numberOfFilters;
    const int neuronPositionX = roughenX(n, this->shapeOfInput[0]);
    const int neuronPositionY = roughenY(n, this->shapeOfInput[0]);

    for (int x = 0; x < this->sizeOfFilterMatrix; ++x)
    {
        for (int y = 0; y < this->sizeOfFilterMatrix; ++y)
        {
            for (int z = 0; z < this->shapeOfInput[2]; ++z)
            {
                const int i = flatten(neuronPositionX + x, neuronPositionY + y, z, this->shapeOfInput[0], this->shapeOfInput[1]);
                neuronInputs.push_back(inputs[i]);
            }
        }
    }
    return neuronInputs;
}

void Convolution2D::insertBackOutputForNeuron(const int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const
{
    const int n = neuronNumber % this->getNumberOfNeurons() / this->numberOfFilters;
    const int neuronPositionX = roughenX(n, this->shapeOfInput[0]);
    const int neuronPositionY = roughenY(n, this->shapeOfInput[0]);

    for (int x = 0; x < this->sizeOfFilterMatrix; ++x)
    {
        for (int y = 0; y < this->sizeOfFilterMatrix; ++y)
        {
            for (int z = 0; z < this->shapeOfInput[2]; ++z)
            {
                const int i = flatten(neuronPositionX + x, neuronPositionY + y, z, this->shapeOfInput[0], this->shapeOfInput[1]);
                const int j = flatten(x, y, z, this->sizeOfFilterMatrix, this->sizeOfFilterMatrix);
                errors[i] += error[j];
            }
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
