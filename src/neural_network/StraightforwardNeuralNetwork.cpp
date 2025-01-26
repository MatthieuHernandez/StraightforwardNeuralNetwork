#include "StraightforwardNeuralNetwork.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "ExtendedExpection.hpp"
#include "NeuralNetworkVisualization.hpp"

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

void StraightforwardNeuralNetwork::startTrainingAsync(Data& data, int batchSize, int evaluationFrequency)
{
    this->validData(data, batchSize);
    this->stopTrainingAsync();
    tools::log<complete>("Start a new thread");
    this->thread = std::thread(&StraightforwardNeuralNetwork::trainSync, this, std::ref(data), Wait(), batchSize,
                               evaluationFrequency);
}

void StraightforwardNeuralNetwork::train(Data& data, Wait wait, int batchSize, int evaluationFrequency)
{
    this->validData(data, batchSize);
    if (!this->isIdle)
    {
        throw std::runtime_error("Neural network is already training");
    }
    this->stopTrainingAsync();
    this->trainSync(data, wait, batchSize, evaluationFrequency);
}

void StraightforwardNeuralNetwork::trainSync(Data& data, Wait wait, const int batchSize, const int evaluationFrequency)
{
    tools::log<minimal>("Start training");
    wait.startClock();
    this->epoch = 0;
    this->numberOfTrainingsBetweenTwoEvaluations = static_cast<int>(data.set.training.size);
    this->wantToStopTraining = false;
    this->isIdle = false;
    if (evaluationFrequency > 0) this->evaluate(data, wait);
    for (this->epoch = 1; this->continueTraining(wait); this->epoch++)
    {
        data.shuffle();
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

            if (data.needToLearnOnTrainingData(this->index))
            {
                this->trainOnce(data.getTrainingData(this->index, batchSize),
                                data.getTrainingOutputs(this->index, batchSize),
                                data.isFirstTrainingDataOfTemporalSequence(this->index));
            }
            else
            {
                this->outputForTraining(data.getTrainingData(this->index, batchSize),
                                        data.isFirstTrainingDataOfTemporalSequence(this->index));
            }
            this->logInProgress<minimal>(wait, data, setType::training);
        }
        if (evaluationFrequency > 0 && this->epoch % evaluationFrequency == 0) this->evaluate(data, wait);
    }
    this->resetTrainingValues();
    tools::log<minimal>("Stop training");
}

void StraightforwardNeuralNetwork::evaluate(const Data& data)
{
    auto wait = Wait();
    this->evaluate(data, wait);
}

void StraightforwardNeuralNetwork::evaluate(const Data& data, Wait& wait)
{
    this->startTesting();
    for (this->index = 0; this->index < data.set.testing.size; this->index++)
    {
        if (this->hasNan())
        {
            this->setResultsAsNan();
            this->resetTrainingValues();
            return;
        }

        if (data.needToEvaluateOnTestingData(this->index))
        {
            this->evaluateOnce(data);
        }
        else
        {
            this->output(data.getTestingData(this->index), data.isFirstTestingDataOfTemporalSequence(this->index));
        }
        this->logInProgress<minimal>(wait, data, setType::testing);
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

inline void StraightforwardNeuralNetwork::evaluateOnce(const Data& data)
{
    switch (data.typeOfProblem)
    {
        case problem::classification:
            this->evaluateOnceForClassification(data.getTestingData(this->index), data.getTestingLabel(this->index),
                                                data.getSeparator(),
                                                data.isFirstTestingDataOfTemporalSequence(this->index));
            break;
        case problem::multipleClassification:
            this->evaluateOnceForMultipleClassification(data.getTestingData(this->index),
                                                        data.getTestingOutputs(this->index), data.getSeparator(),
                                                        data.isFirstTestingDataOfTemporalSequence(this->index));
            break;
        case problem::regression:
            this->evaluateOnceForRegression(data.getTestingData(this->index), data.getTestingOutputs(this->index),
                                            data.getPrecision(),
                                            data.isFirstTestingDataOfTemporalSequence(this->index));
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

    if (this->hasNan()) tools::log<minimal>("A NaN value has been detected");

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

void StraightforwardNeuralNetwork::validData(const Data& data, int batchSize) const
{
    if (data.numberOfLabels != this->getNumberOfOutputs() || data.sizeOfData != this->getNumberOfInputs())
    {
        throw std::runtime_error("Data has not the same format as the neural network");
    }
    if (batchSize != 1 && data.typeOfTemporal != nature::nonTemporal)
    {
        throw std::runtime_error("Non temporal data cannot have batch size");
    }
    if (batchSize < 1)
    {
        throw std::runtime_error("Wrong batch size");
    }
    if (batchSize > data.set.training.size)
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
    for (size_t l = 0; l < this->layers.size(); ++l)
    {
        const auto filterLayer = dynamic_cast<internal::FilterLayer*>(this->layers[l].get());
        internal::NeuralNetworkVisualization::saveAsBitmap(filterLayer,
                                                           filePath + "_layer_" + std::to_string(l) + ".bmp");
    }
}

void StraightforwardNeuralNetwork::saveData2DAsBitmap(const std::string& filePath, const Data& data, int dataIndex)
{
    internal::NeuralNetworkVisualization::saveAsBitmap(data.getTestingData(dataIndex),
                                                       this->layers[0]->getShapeOfInput(),
                                                       filePath + "_input" + std::to_string(dataIndex) + ".bmp");
}

void StraightforwardNeuralNetwork::saveFilterLayersAsBitmap(const std::string& filePath, const Data& data,
                                                            int dataIndex)
{
    if (!this->isIdle)
    {
        throw std::runtime_error("Filter layers cannot be saved during training, stop training before saving");
    }

    auto outputs = this->getLayerOutputs(data.getTestingData(dataIndex));

    for (size_t l = 0; l < this->layers.size(); ++l)
    {
        const auto filterLayer = dynamic_cast<internal::FilterLayer*>(this->layers[l].get());

        internal::NeuralNetworkVisualization::saveAsBitmap(filterLayer, outputs[l],
                                                           filePath + "_layer_" + std::to_string(l) + ".bmp");
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
