#include "NeuralNetwork.hpp"

#include <boost/serialization/export.hpp>
#include <cmath>
#include <ctime>

#include "layer/LayerFactory.hpp"
#include "layer/LayerModel.hpp"
#include "optimizer/NeuralNetworkOptimizerFactory.hpp"

namespace snn::internal
{
NeuralNetwork::NeuralNetwork(std::vector<snn::LayerModel>& architecture, snn::NeuralNetworkOptimizerModel optimizer)
{
    this->optimizer = NeuralNetworkOptimizerFactory::build(optimizer);
    LayerFactory::build(this->layers, architecture, this->optimizer);
    this->StatisticAnalysis::initialize(this->layers.back()->getNumberOfNeurons());
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
    : StatisticAnalysis(neuralNetwork),
      outputNan(neuralNetwork.outputNan),
      numberOfTraining(neuralNetwork.numberOfTraining),
      optimizer(neuralNetwork.optimizer->clone())
{
    this->layers.reserve(neuralNetwork.layers.size());
    for (const auto& layer : neuralNetwork.layers)
    {
        this->layers.push_back(layer->clone(this->optimizer));
    }
    this->StatisticAnalysis::initialize(this->layers.back()->getNumberOfNeurons());
}

void NeuralNetwork::evaluateOnceForRegression(const std::vector<float>& inputs, const std::vector<float>& desired,
                                              const float precision, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForRegression(outputs, desired, precision);
}

void NeuralNetwork::evaluateOnceForMultipleClassification(const std::vector<float>& inputs,
                                                          const std::vector<float>& desired, const float separator,
                                                          bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForMultipleClassification(outputs, desired, separator);
}

void NeuralNetwork::evaluateOnceForClassification(const std::vector<float>& inputs, const int classNumber,
                                                  const float separator, bool temporalReset)
{
    const auto outputs = this->output(inputs, temporalReset);
    this->StatisticAnalysis::evaluateOnceForClassification(outputs, classNumber, separator);
}

void NeuralNetwork::trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired, bool temporalReset)
{
    this->backpropagationAlgorithm(inputs, desired, temporalReset);
    this->numberOfTraining++;
}

auto NeuralNetwork::output(const std::vector<float>& inputs, bool temporalReset) -> std::vector<float>
{
    auto outputs = layers[0]->output(inputs, temporalReset);
    for (size_t l = 1; l < this->layers.size(); ++l)
    {
        outputs = layers[l]->output(outputs, temporalReset);
    }
    if (std::ranges::any_of(
            outputs, [](const float& v) { return std::fpclassify(v) != FP_NORMAL && std::fpclassify(v) != FP_ZERO; }))
    {
        this->outputNan = true;
    }
    return outputs;
}

auto NeuralNetwork::outputForTraining(const std::vector<float>& inputs, bool temporalReset) -> std::vector<float>
{
    auto outputs = layers[0]->outputForTraining(inputs, temporalReset);

    for (size_t l = 1; l < this->layers.size(); ++l)
    {
        outputs = layers[l]->outputForTraining(outputs, temporalReset);
    }

    if (std::ranges::any_of(
            outputs, [](const float& v) { return std::fpclassify(v) != FP_NORMAL && std::fpclassify(v) != FP_ZERO; }))
    {
        this->outputNan = true;
    }

    return outputs;
}

void NeuralNetwork::backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired,
                                             bool temporalReset)
{
    const auto outputs = this->outputForTraining(inputs, temporalReset);
    if (this->outputNan)
    {
        return;
    }
    auto errors = this->calculateError(outputs, desired);

    for (size_t l = this->layers.size() - 1; l > 0; --l)
    {
        errors = layers[l]->backOutput(errors);
    }
    layers[0]->train(errors);
}

inline auto NeuralNetwork::calculateError(const std::vector<float>& outputs, const std::vector<float>& desired) const
    -> std::vector<float>
{
    std::vector<float> errors(this->layers.back()->getNumberOfNeurons(), 0);
    for (size_t n = 0; n < errors.size(); ++n)
    {
        if (std::fpclassify(desired[n]) != FP_NORMAL && std::fpclassify(desired[n]) != FP_ZERO)
        {
            errors[n] = 0.0F;
        }
        else
        {
            errors[n] = 2.0F * (desired[n] - outputs[n]);
        }
    }
    return errors;
}

auto NeuralNetwork::getLayerOutputs(const std::vector<float>& inputs) -> snn::vector2D<float>
{
    snn::vector2D<float> filterOutputs;
    auto outputs = layers[0]->output(inputs, true);
    filterOutputs.push_back(outputs);

    for (size_t l = 1; l < this->layers.size(); ++l)
    {
        outputs = layers[l]->output(outputs, true);
        filterOutputs.push_back(outputs);
    }
    return filterOutputs;
}

auto NeuralNetwork::hasNan() const -> bool { return this->outputNan; }

auto NeuralNetwork::getNumberOfTraining() const -> int64_t { return this->numberOfTraining; }

auto NeuralNetwork::getNumberOfLayers() const -> int { return static_cast<int>(this->layers.size()); }

auto NeuralNetwork::getNumberOfInputs() const -> int { return this->layers[0]->getNumberOfInputs(); }

auto NeuralNetwork::getNumberOfOutputs() const -> int { return this->layers.back()->getNumberOfNeurons(); }

auto NeuralNetwork::getNumberOfNeurons() const -> int
{
    int sum = 0;
    for (auto& layer : this->layers)
    {
        sum += layer->getNumberOfNeurons();
    }
    return sum;
}

auto NeuralNetwork::getNumberOfParameters() const -> int
{
    int sum = 0;
    for (const auto& layer : this->layers) sum += layer->getNumberOfParameters();
    return sum;
}

auto NeuralNetwork::isValid() const -> errorType
{
    // TODO(matth): rework isValid
    constexpr int big_image_size = 1920 * 1080;  // 2073600
    if (this->getNumberOfInputs() < 1 || this->getNumberOfInputs() > big_image_size)
    {
        return errorType::neuralNetworkInputTooLarge;
    }
    constexpr int lot_of_layers = 100;
    if (this->getNumberOfLayers() < 1 || this->getNumberOfLayers() > lot_of_layers)
    {
        return errorType::neuralNetworkTooMuchLayers;
    }
    auto err = this->optimizer->isValid();
    if (err != errorType::noError)
    {
        return err;
    }
    for (const auto& layer : this->layers)
    {
        err = layer->isValid();
        if (err != errorType::noError)
        {
            return err;
        }
    }
    return errorType::noError;
}

auto NeuralNetwork::operator==(const NeuralNetwork& neuralNetwork) const -> bool
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

auto NeuralNetwork::operator!=(const NeuralNetwork& neuralNetwork) const -> bool { return !(*this == neuralNetwork); }
}  // namespace snn::internal
