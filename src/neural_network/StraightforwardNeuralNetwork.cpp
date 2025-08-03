#include "StraightforwardNeuralNetwork.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ExtendedExpection.hpp"
#include "NeuralNetworkVisualization.hpp"
#include "layer/LayerModel.hpp"

namespace snn
{
StraightforwardNeuralNetwork::~StraightforwardNeuralNetwork() { this->stopTrainingAsync(); }

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(std::vector<LayerModel> architecture,
                                                           NeuralNetworkOptimizerModel optimizer)
    : NeuralNetwork(architecture, optimizer)
{
    const auto err = this->isValid();
    if (err != errorType::noError)
    {
        const std::string message =
            std::string("Error ") + tools::toString(err) + ": Wrong parameter in the creation of neural networks";
        throw std::runtime_error(message);
    }
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork)
    : NeuralNetwork(neuralNetwork)
{
    if (!this->isIdle)
    {
        throw std::runtime_error("StraightforwardNeuralNetwork must be idle to be copy");
    }
    this->autoSaveFilePath = neuralNetwork.autoSaveFilePath;
    this->autoSaveWhenBetter = neuralNetwork.autoSaveWhenBetter;
    this->index = neuralNetwork.index;
    this->epoch = neuralNetwork.epoch;
    this->numberOfTrainingsBetweenTwoEvaluations = neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
}

auto StraightforwardNeuralNetwork::computeOutput(const std::vector<float>& inputs, bool temporalReset)
    -> std::vector<float>
{
    return this->output(inputs, temporalReset);
}

auto StraightforwardNeuralNetwork::computeCluster(const std::vector<float>& inputs, bool temporalReset) -> int
{
    const auto outputs = this->output(inputs, temporalReset);
    float maxOutputValue = -2;
    int maxOutputIndex = -1;
    for (int i = 0; i < static_cast<int>(outputs.size()); i++)
    {
        if (maxOutputValue < outputs[i])
        {
            maxOutputValue = outputs[i];
            maxOutputIndex = i;
        }
    }
    return maxOutputIndex;
}

auto StraightforwardNeuralNetwork::isTraining() const -> bool { return !this->isIdle; }

void StraightforwardNeuralNetwork::startTrainingAsync(Dataset& dataset, int batchSize, int evaluationFrequency)
{
    this->validData(dataset, batchSize);
    this->stopTrainingAsync();
    tools::log<complete>("Start a new thread");
    this->thread = std::thread(&StraightforwardNeuralNetwork::trainSync, this, std::ref(dataset), Wait(), batchSize,
                               evaluationFrequency);
}

void StraightforwardNeuralNetwork::train(Dataset& dataset, Wait wait, int batchSize, int evaluationFrequency)
{
    this->validData(dataset, batchSize);
    if (!this->isIdle)
    {
        throw std::runtime_error("Neural network is already training");
    }
    this->stopTrainingAsync();
    this->trainSync(dataset, wait, batchSize, evaluationFrequency);
}

void StraightforwardNeuralNetwork::trainSync(Dataset& dataset, Wait wait, const int batchSize,
                                             const int evaluationFrequency)
{
    tools::log<minimal>("Start training");
    wait.startClock();
    this->epoch = 0;
    this->numberOfTrainingsBetweenTwoEvaluations = dataset.data.training.size;
    this->wantToStopTraining = false;
    this->isIdle = false;
    if (evaluationFrequency > 0)
    {
        this->evaluate(dataset, wait);
    }
    for (this->epoch = 1; this->continueTraining(wait); this->epoch++)
    {
        dataset.shuffle();
        for (this->index = 0;
             index + batchSize <= this->numberOfTrainingsBetweenTwoEvaluations && this->continueTraining(wait);
             this->index += batchSize)
        {
            if (this->hasNan())
            {
                this->setResultsAsNan();
                this->resetTrainingValues();
                return;
            }

            if (dataset.needToLearnOnTrainingData(this->index))
            {
                this->trainOnce(dataset.getTrainingData(this->index, batchSize),
                                dataset.getTrainingOutputs(this->index, batchSize), {},
                                dataset.isFirstTrainingDataOfTemporalSequence(this->index));
            }
            else
            {
                this->outputForTraining(dataset.getTrainingData(this->index, batchSize),
                                        dataset.isFirstTrainingDataOfTemporalSequence(this->index));
            }
            this->logInProgress<minimal>(wait, dataset, setType::training);
        }
        if (evaluationFrequency > 0 && this->epoch % evaluationFrequency == 0)
        {
            this->evaluate(dataset, wait);
        }
    }
    this->resetTrainingValues();
    tools::log<minimal>("Stop training");
}

void StraightforwardNeuralNetwork::evaluate(const Dataset& dataset)
{
    auto wait = Wait();
    wait.startClock();
    this->evaluate(dataset, wait);
}

void StraightforwardNeuralNetwork::evaluate(const Dataset& dataset, Wait& wait)
{
    this->startTesting();
    for (this->index = 0; this->index < dataset.data.testing.size; this->index++)
    {
        if (this->hasNan())
        {
            this->setResultsAsNan();
            this->resetTrainingValues();
            return;
        }

        if (dataset.needToEvaluateOnTestingData(this->index))
        {
            this->evaluateOnce(dataset);
        }
        else
        {
            this->output(dataset.getTestingData(this->index),
                         dataset.isFirstTestingDataOfTemporalSequence(this->index));
        }
        this->logInProgress<minimal>(wait, dataset, setType::testing);
    }
    this->stopTesting();
    if (this->autoSaveWhenBetter && this->globalClusteringRateIsBetterThanMax)
    {
        this->saveSync(autoSaveFilePath);
        this->logAccuracy<minimal>(wait, true);
    }
    else
    {
        this->logAccuracy<minimal>(wait, false);
    }
}

inline void StraightforwardNeuralNetwork::evaluateOnce(const Dataset& dataset)
{
    switch (dataset.typeOfProblem)
    {
        case problem::classification:
            this->evaluateOnceForClassification(dataset.getTestingData(this->index),
                                                dataset.getTestingLabel(this->index), dataset.getSeparator(),
                                                dataset.isFirstTestingDataOfTemporalSequence(this->index));
            break;
        case problem::multipleClassification:
            this->evaluateOnceForMultipleClassification(dataset.getTestingData(this->index),
                                                        dataset.getTestingOutputs(this->index), dataset.getSeparator(),
                                                        dataset.isFirstTestingDataOfTemporalSequence(this->index));
            break;
        case problem::regression:
            this->evaluateOnceForRegression(dataset.getTestingData(this->index), dataset.getTestingOutputs(this->index),
                                            dataset.getPrecision(),
                                            dataset.isFirstTestingDataOfTemporalSequence(this->index));
            break;
        default:
            throw NotImplementedException();
    }
}

void StraightforwardNeuralNetwork::stopTrainingAsync()
{
    this->wantToStopTraining = true;
    if (this->thread.joinable())
    {
        tools::log<minimal>("Stop training");
        this->thread.join();
        tools::log<complete>("Thread closed");
    }
}

void StraightforwardNeuralNetwork::resetTrainingValues()
{
    this->index = 0;
    this->wantToStopTraining = false;
    this->isIdle = true;
}

inline auto StraightforwardNeuralNetwork::continueTraining(Wait wait) const -> bool
{
    const auto epochs = this->getCurrentEpoch();
    const auto accuracy = this->getGlobalClusteringRate();
    const auto mae = this->getMeanAbsoluteError();

    if (this->hasNan())
    {
        tools::log<minimal>("A NaN value has been detected");
    }

    return !this->wantToStopTraining && !wait.isOver(epochs, accuracy, mae) && !this->hasNan();
}

void StraightforwardNeuralNetwork::waitFor(Wait wait) const
{
    wait.startClock();
    while (!this->hasNan())
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1ms);

        const auto epochs = this->getCurrentEpoch();
        const auto accuracy = this->getGlobalClusteringRate();
        const auto mae = this->getMeanAbsoluteError();

        if (wait.isOver(epochs, accuracy, mae))
        {
            break;
        }
    }
}

auto StraightforwardNeuralNetwork::isValid() const -> errorType { return this->NeuralNetwork::isValid(); }

void StraightforwardNeuralNetwork::validData(const Dataset& dataset, int batchSize) const
{
    if (dataset.numberOfLabels != this->getNumberOfOutputs() || dataset.sizeOfData != this->getNumberOfInputs())
    {
        throw std::runtime_error("Data has not the same format as the neural network");
    }
    if (batchSize != 1 && dataset.typeOfTemporal != nature::nonTemporal)
    {
        throw std::runtime_error("Non temporal data cannot have batch size");
    }
    if (batchSize < 1)
    {
        throw std::runtime_error("Wrong batch size");
    }
    if (batchSize > dataset.data.training.size)
    {
        throw std::runtime_error("Batch size is too large");
    }
}

void StraightforwardNeuralNetwork::saveAs(const std::string& filePath)
{
    if (!this->isIdle)
    {
        throw std::runtime_error(
            "Neural network cannot be saved during training, stop training before saving or use auto save");
    }
    saveSync(filePath);
}

void StraightforwardNeuralNetwork::saveFeatureMapsAsBitmap(const std::string& filePath)
{
    if (!this->isIdle)
    {
        throw std::runtime_error("Filter layers cannot be saved during training, stop training before saving");
    }
    for (size_t layer = 0; layer < this->layers.size(); ++layer)
    {
        auto* filterLayer = dynamic_cast<internal::FilterLayer*>(this->layers[layer].get());
        internal::NeuralNetworkVisualization::saveAsBitmap(filterLayer,
                                                           filePath + "_layer_" + std::to_string(layer) + ".bmp");
    }
}

void StraightforwardNeuralNetwork::saveData2DAsBitmap(const std::string& filePath, const Dataset& dataset,
                                                      int dataIndex)
{
    internal::NeuralNetworkVisualization::saveAsBitmap(dataset.getTestingData(dataIndex),
                                                       this->layers[0]->getShapeOfInput(),
                                                       filePath + "_input" + std::to_string(dataIndex) + ".bmp");
}

void StraightforwardNeuralNetwork::saveFilterLayersAsBitmap(const std::string& filePath, const Dataset& dataset,
                                                            int dataIndex)
{
    if (!this->isIdle)
    {
        throw std::runtime_error("Filter layers cannot be saved during training, stop training before saving");
    }

    auto outputs = this->getLayerOutputs(dataset.getTestingData(dataIndex));

    for (size_t layer = 0; layer < this->layers.size(); ++layer)
    {
        auto* filterLayer = dynamic_cast<internal::FilterLayer*>(this->layers[layer].get());
        internal::NeuralNetworkVisualization::saveAsBitmap(filterLayer, outputs[layer],
                                                           filePath + "_layer_" + std::to_string(layer) + ".bmp");
    }
}

void StraightforwardNeuralNetwork::saveSync(const std::string& filePath)
{
    this->autoSaveFilePath = filePath;
    std::ofstream ofs(filePath);
    boost::archive::text_oarchive archive(ofs);
    archive << this;
}

auto StraightforwardNeuralNetwork::loadFrom(const std::string& filePath) -> StraightforwardNeuralNetwork&
{
    StraightforwardNeuralNetwork* neuralNetwork = nullptr;
    std::ifstream ifs(filePath);
    boost::archive::text_iarchive archive(ifs);
    archive >> neuralNetwork;
    neuralNetwork->resetLearningVariables();
    return *neuralNetwork;
}

auto StraightforwardNeuralNetwork::summary() const -> std::string
{
    std::stringstream summary;
    summary << "============================================================\n";
    summary << "| SNN Model Summary                                        |\n";
    summary << "============================================================\n";
    summary << " Name:       " << this->autoSaveFilePath << '\n';
    summary << " Parameters: " << this->getNumberOfParameters() << '\n';
    summary << " Epochs:     " << this->epoch << '\n';
    summary << " Trainnig:   " << this->getNumberOfTraining() << '\n';
    summary << "============================================================\n";
    summary << "| Layers                                                   |\n";
    summary << "============================================================\n";
    for (const auto& layer : this->layers)
    {
        summary << layer->summary();
    }
    summary << "============================================================\n";
    summary << "|  Optimizer                                               |\n";
    summary << "============================================================\n";
    summary << optimizer->summary();
    summary << "============================================================\n";
    return summary.str();
}

auto StraightforwardNeuralNetwork::operator==(const StraightforwardNeuralNetwork& neuralNetwork) const -> bool
{
    return this->NeuralNetwork::operator==(neuralNetwork) && this->autoSaveFilePath == neuralNetwork.autoSaveFilePath &&
           this->autoSaveWhenBetter == neuralNetwork.autoSaveWhenBetter &&
           this->wantToStopTraining == neuralNetwork.wantToStopTraining && this->index == neuralNetwork.index &&
           this->isIdle == neuralNetwork.isIdle && this->epoch == neuralNetwork.epoch &&
           this->numberOfTrainingsBetweenTwoEvaluations == neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
}

auto StraightforwardNeuralNetwork::operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const -> bool
{
    return !(*this == neuralNetwork);
}
}  // namespace snn
