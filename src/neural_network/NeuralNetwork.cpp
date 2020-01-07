#include <algorithm>
#include <ctime>
#include <boost/serialization/export.hpp>
#include "NeuralNetwork.hpp"
#include "layer/LayerModel.hpp"
#include "../tools/ExtendedExpection.hpp"
#include "layer/LayerFactory.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(NeuralNetwork)

bool NeuralNetwork::isTheFirst = true;

void NeuralNetwork::initialize()
{
    srand(static_cast<int>(time(nullptr)));
    rand();
    ActivationFunction::initialize();
    isTheFirst = false;
}

NeuralNetwork::NeuralNetwork(int numberOfInputs, vector<LayerModel>& models)
    :StatisticAnalysis(models.back().numberOfNeurons)
{
    if (isTheFirst)
        this->initialize();
    LayerFactory::build(this->layers, numberOfInputs, models, &this->learningRate, &this->momentum);
    
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
    : StatisticAnalysis(neuralNetwork.getNumberOfOutputs()),
    maxOutputIndex(neuralNetwork.maxOutputIndex),
    learningRate(neuralNetwork.learningRate),
    momentum(neuralNetwork.momentum)
{
    this->layers.reserve(neuralNetwork.layers.size());
    for (const auto& layer : neuralNetwork.layers)
        this->layers.push_back(layer->clone());
}

inline
int NeuralNetwork::getNumberOfLayers() const
{
    return this->layers.size();
}

int NeuralNetwork::getNumberOfInputs() const
{
    return this->layers[0]->getNumberOfInputs();
}

int NeuralNetwork::getNumberOfOutputs() const
{
    return this->layers.back()->getNumberOfNeurons();
}

int NeuralNetwork::isValid() const
{
    //TODO: rework isValid
    if (this->getNumberOfInputs() < 1 
    || this->getNumberOfInputs() > 2073600) // 1920 * 1080
        return 101;

    if (this->getNumberOfLayers() < 1 
    || this->getNumberOfLayers() > 1000)
        return 102;

    if (this->learningRate <= 0.0f || this->learningRate >= 1.0f)
        return 103;

    if (this->momentum < 0.0f || this->momentum > 1.0f)
        return 104;

    for (auto& layer : this->layers)
    {
        int err = layer->isValid();
        if(err != 0)
            return err;
    }
    return 0;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
    return this->maxOutputIndex == neuralNetwork.maxOutputIndex
        && this->learningRate == neuralNetwork.learningRate
        && this->momentum == neuralNetwork.momentum
        && [=] () {
            for (int l = 0; l < this->layers.size(); l++)
            {
                if (*this->layers[l] != *neuralNetwork.layers[l])
                    return false;
            }
            return true;
        }();
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const
{
    return !(*this == neuralNetwork);
}
