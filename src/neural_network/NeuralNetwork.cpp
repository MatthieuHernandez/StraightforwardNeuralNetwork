#include "NeuralNetwork.hpp"

#include <boost/serialization/export.hpp>
#include <cmath>
#include <ctime>

#include "layer/LayerFactory.hpp"
#include "layer/LayerModel.hpp"
#include "optimizer/NeuralNetworkOptimizerFactory.hpp"

using namespace std;
using namespace snn;
using namespace internal;

NeuralNetwork::NeuralNetwork(vector<LayerModel>& architecture, NeuralNetworkOptimizerModel optimizer)
{
    this->optimizer = NeuralNetworkOptimizerFactory::build(optimizer);
    LayerFactory::build(this->layers, architecture, this->optimizer);
    this->StatisticAnalysis::initialize(this->layers.back()->getNumberOfNeurons());
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
    : StatisticAnalysis(neuralNetwork),
      optimizer(neuralNetwork.optimizer->clone())
{
    this->outputNan = neuralNetwork.outputNan;
    this->numberOfTraining = neuralNetwork.numberOfTraining;
    this->layers.reserve(neuralNetwork.layers.size());
    for (const auto& layer : neuralNetwork.layers) this->layers.push_back(layer->clone(this->optimizer));
    this->StatisticAnalysis::initialize(this->layers.back()->getNumberOfNeurons());
}

void NeuralNetwork::evaluateOnceForRegression(const vector<float>& inputs, const vector<float>& desired,
                                              const float precision, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForRegression(outputs, desired, precision);
}

void NeuralNetwork::evaluateOnceForMultipleClassification(const vector<float>& inputs, const vector<float>& desired,
                                                          const float separator, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForMultipleClassification(outputs, desired, separator);
}

void NeuralNetwork::evaluateOnceForClassification(const vector<float>& inputs, const int classNumber,
                                                  const float separator, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForClassification(outputs, classNumber, separator);
}

void NeuralNetwork::trainOnce(const vector<float>& inputs, const vector<float>& desired, bool temporalReset)
{
    this->backpropagationAlgorithm(inputs, desired, temporalReset);
    this->numberOfTraining++;
}

vector<float> NeuralNetwork::output(const vector<float>& inputs, bool temporalReset)
{
    auto outputs = layers[0]->output(inputs, temporalReset);

    for (size_t l = 1; l < this->layers.size(); ++l) outputs = layers[l]->output(outputs, temporalReset);

    if (ranges::any_of(outputs, [](const float& v) { return fpclassify(v) != FP_NORMAL && fpclassify(v) != FP_ZERO; }))
        this->outputNan = true;

    return outputs;
}

std::vector<float> NeuralNetwork::outputForTraining(const std::vector<float>& inputs, bool temporalReset)
{
    auto outputs = layers[0]->outputForTraining(inputs, temporalReset);

    for (size_t l = 1; l < this->layers.size(); ++l) outputs = layers[l]->outputForTraining(outputs, temporalReset);

    if (ranges::any_of(outputs, [](const float& v) { return fpclassify(v) != FP_NORMAL && fpclassify(v) != FP_ZERO; }))
        this->outputNan = true;

    return outputs;
}

void NeuralNetwork::backpropagationAlgorithm(const vector<float>& inputs, const vector<float>& desired,
                                             bool temporalReset)
{
    const auto outputs = this->outputForTraining(inputs, temporalReset);
    if (this->outputNan) return;
    auto errors = this->calculateError(outputs, desired);

    for (size_t l = this->layers.size() - 1; l > 0; --l) errors = layers[l]->backOutput(errors);
    layers[0]->train(errors);
}

inline vector<float> NeuralNetwork::calculateError(const vector<float>& outputs, const vector<float>& desired) const
{
    vector<float> errors(this->layers.back()->getNumberOfNeurons(), 0);
    for (size_t n = 0; n < errors.size(); ++n)
    {
        if (fpclassify(desired[n]) != FP_NORMAL && fpclassify(desired[n]) != FP_ZERO)
            errors[n] = 0.0f;
        else
            errors[n] = 2.0f * (desired[n] - outputs[n]);
    }
    return errors;
}

vector2D<float> NeuralNetwork::getLayerOutputs(const vector<float>& inputs)
{
    vector2D<float> filterOutputs;
    auto outputs = layers[0]->output(inputs, true);
    filterOutputs.push_back(outputs);

    for (size_t l = 1; l < this->layers.size(); ++l)
    {
        outputs = layers[l]->output(outputs, true);
        filterOutputs.push_back(outputs);
    }
    return filterOutputs;
}

bool NeuralNetwork::hasNan() const { return this->outputNan; }

int64_t NeuralNetwork::getNumberOfTraining() const { return this->numberOfTraining; }

int NeuralNetwork::getNumberOfLayers() const { return (int)this->layers.size(); }

int NeuralNetwork::getNumberOfInputs() const { return this->layers[0]->getNumberOfInputs(); }

int NeuralNetwork::getNumberOfOutputs() const { return this->layers.back()->getNumberOfNeurons(); }

int NeuralNetwork::getNumberOfNeurons() const
{
    int sum = 0;
    for (auto& layer : this->layers)
    {
        sum += layer->getNumberOfNeurons();
    }
    return sum;
}

int NeuralNetwork::getNumberOfParameters() const
{
    int sum = 0;
    for (const auto& layer : this->layers) sum += layer->getNumberOfParameters();
    return sum;
}

int NeuralNetwork::isValid() const
{
    // TODO: rework isValid
    if (this->getNumberOfInputs() < 1 || this->getNumberOfInputs() > 2073600)  // 1920 * 1080
        return 101;

    if (this->getNumberOfLayers() < 1 || this->getNumberOfLayers() > 1000) return 102;

    int err = this->optimizer->isValid();
    if (err != 0) return err;

    for (const auto& layer : this->layers)
    {
        err = layer->isValid();
        if (err != 0) return err;
    }
    return 0;
}

bool NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const
{
    return *this->optimizer == *neuralNetwork.optimizer && this->layers.size() == neuralNetwork.layers.size() &&
           [this, &neuralNetwork]()
    {
        for (size_t l = 0; l < this->layers.size(); ++l)
        {
            if (*this->layers[l] != *neuralNetwork.layers[l]) return false;
        }
        return true;
    }();
}

bool NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const { return !(*this == neuralNetwork); }
