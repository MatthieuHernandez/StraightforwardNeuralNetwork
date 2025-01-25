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

using namespace std;
using namespace snn;
using namespace internal;
using namespace tools;

StraightforwardNeuralNetwork::~StraightforwardNeuralNetwork() { this->stopTrainingAsync(); }

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(vector<LayerModel> architecture,
                                                           NeuralNetworkOptimizerModel optimizer)
    : NeuralNetwork(architecture, optimizer)
{
    const auto err = this->isValid();
    if (err != ErrorType::noError)
    {
        const string message =
            string("Error ") + tools::toString(err) + ": Wrong parameter in the creation of neural networks";
        throw runtime_error(message);
    }
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork)
    : NeuralNetwork(neuralNetwork)
{
    if (!this->isIdle) throw runtime_error("StraightforwardNeuralNetwork must be idle to be copy");
    this->autoSaveFilePath = neuralNetwork.autoSaveFilePath;
    this->autoSaveWhenBetter = neuralNetwork.autoSaveWhenBetter;
    this->index = neuralNetwork.index;
    this->epoch = neuralNetwork.epoch;
    this->numberOfTrainingsBetweenTwoEvaluations = neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
}

auto StraightforwardNeuralNetwork::computeOutput(const vector<float>& inputs, bool temporalReset) -> vector<float>
{
    return this->output(inputs, temporalReset);
}

auto StraightforwardNeuralNetwork::computeCluster(const vector<float>& inputs, bool temporalReset) -> int
{
    const auto outputs = this->output(inputs, temporalReset);
    float maxOutputValue = -2;
    int maxOutputIndex = -1;
    for (int i = 0; i < (int)outputs.size(); i++)
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
    log<complete>("Start a new thread");
    this->thread =
        std::thread(&StraightforwardNeuralNetwork::trainSync, this, ref(data), Wait(), batchSize, evaluationFrequency);
}

void StraightforwardNeuralNetwork::train(Data& data, Wait wait, int batchSize, int evaluationFrequency)
{
    this->validData(data, batchSize);
    if (!this->isIdle) throw runtime_error("Neural network is already training");
    this->stopTrainingAsync();
    this->trainSync(data, wait, batchSize, evaluationFrequency);
}

void StraightforwardNeuralNetwork::trainSync(Data& data, Wait wait, const int batchSize, const int evaluationFrequency)
{
    log<minimal>("Start training");
    wait.startClock();
    this->epoch = 0;
    this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
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
                this->trainOnce(data.getTrainingData(this->index, batchSize),
                                data.getTrainingOutputs(this->index, batchSize),
                                data.isFirstTrainingDataOfTemporalSequence(this->index));
            else
                this->outputForTraining(data.getTrainingData(this->index, batchSize),
                                        data.isFirstTrainingDataOfTemporalSequence(this->index));
            this->logInProgress<minimal>(wait, data, training);
        }
        if (evaluationFrequency > 0 && this->epoch % evaluationFrequency == 0) this->evaluate(data, wait);
    }
    this->resetTrainingValues();
    log<minimal>("Stop training");
}

void StraightforwardNeuralNetwork::evaluate(const Data& data)
{
    auto wait = Wait();
    this->evaluate(data, wait);
}

void StraightforwardNeuralNetwork::evaluate(const Data& data, Wait& wait)
{
    this->startTesting();
    for (this->index = 0; this->index < data.sets[testing].size; this->index++)
    {
        if (this->hasNan())
        {
            this->setResultsAsNan();
            this->resetTrainingValues();
            return;
        }

        if (data.needToEvaluateOnTestingData(this->index))
            this->evaluateOnce(data);
        else
            this->output(data.getTestingData(this->index), data.isFirstTestingDataOfTemporalSequence(this->index));
        this->logInProgress<minimal>(wait, data, testing);
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
        log<minimal>("Stop training");
        this->thread.join();
        log<complete>("Thread closed");
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

    if (this->hasNan()) log<minimal>("A NaN value has been detected");

    return !this->wantToStopTraining && !wait.isOver(epochs, accuracy, mae) && !this->hasNan();
}

void StraightforwardNeuralNetwork::waitFor(Wait wait) const
{
    wait.startClock();
    while (!this->hasNan())
    {
        this_thread::sleep_for(1ms);

        const auto epochs = this->getCurrentEpoch();
        const auto accuracy = this->getGlobalClusteringRate();
        const auto mae = this->getMeanAbsoluteError();

        if (wait.isOver(epochs, accuracy, mae)) break;
    }
}

auto StraightforwardNeuralNetwork::isValid() const -> ErrorType { return this->NeuralNetwork::isValid(); }

void StraightforwardNeuralNetwork::validData(const Data& data, int batchSize) const
{
    if (data.numberOfLabels != this->getNumberOfOutputs() || data.sizeOfData != this->getNumberOfInputs())
        throw runtime_error("Data has not the same format as the neural network");
    if (batchSize != 1 && data.typeOfTemporal != nature::nonTemporal)
        throw runtime_error("Non temporal data cannot have batch size");
    if (batchSize < 1) throw runtime_error("Wrong batch size");
    if (batchSize > data.sets[training].size) throw runtime_error("Batch size is too large");
}

void StraightforwardNeuralNetwork::saveAs(const string filePath)
{
    if (!this->isIdle)
        throw runtime_error(
            "Neural network cannot be saved during training, stop training before saving or use auto save");
    saveSync(filePath);
}

void StraightforwardNeuralNetwork::saveFeatureMapsAsBitmap(string filePath)
{
    if (!this->isIdle)
        throw runtime_error("Filter layers cannot be saved during training, stop training before saving");
    for (size_t l = 0; l < this->layers.size(); ++l)
    {
        const auto filterLayer = dynamic_cast<FilterLayer*>(this->layers[l].get());
        NeuralNetworkVisualization::saveAsBitmap(filterLayer, filePath + "_layer_" + to_string(l) + ".bmp");
    }
}

void StraightforwardNeuralNetwork::saveData2DAsBitmap(string filePath, const Data& data, int dataIndex)
{
    NeuralNetworkVisualization::saveAsBitmap(data.getTestingData(dataIndex), this->layers[0]->getShapeOfInput(),
                                             filePath + "_input" + to_string(dataIndex) + ".bmp");
}

void StraightforwardNeuralNetwork::saveFilterLayersAsBitmap(string filePath, const Data& data, int dataIndex)
{
    if (!this->isIdle)
        throw runtime_error("Filter layers cannot be saved during training, stop training before saving");

    auto outputs = this->getLayerOutputs(data.getTestingData(dataIndex));

    for (size_t l = 0; l < this->layers.size(); ++l)
    {
        const auto filterLayer = dynamic_cast<FilterLayer*>(this->layers[l].get());

        NeuralNetworkVisualization::saveAsBitmap(filterLayer, outputs[l], filePath + "_layer_" + to_string(l) + ".bmp");
    }
}

void StraightforwardNeuralNetwork::saveSync(const string filePath)
{
    this->autoSaveFilePath = filePath;
    ofstream ofs(filePath);
    boost::archive::text_oarchive archive(ofs);
    archive << this;
}

auto StraightforwardNeuralNetwork::loadFrom(const string filePath) -> StraightforwardNeuralNetwork&
{
    StraightforwardNeuralNetwork* neuralNetwork = nullptr;
    ifstream ifs(filePath);
    boost::archive::text_iarchive archive(ifs);
    archive >> neuralNetwork;
    return *neuralNetwork;
}

auto StraightforwardNeuralNetwork::summary() const -> string
{
    stringstream ss;
    ss << "============================================================" << endl;
    ss << "| SNN Model Summary                                        |" << endl;
    ss << "============================================================" << endl;
    ss << " Name:       " << this->autoSaveFilePath << endl;
    ss << " Parameters: " << this->getNumberOfParameters() << endl;
    ss << " Epochs:     " << this->epoch << endl;
    ss << " Trainnig:   " << this->getNumberOfTraining() << endl;
    ss << "============================================================" << endl;
    ss << "| Layers                                                   |" << endl;
    ss << "============================================================" << endl;
    for (auto& layer : this->layers) ss << layer->summary();
    ss << "============================================================" << endl;
    ss << "|  Optimizer                                               |" << endl;
    ss << "============================================================" << endl;
    ss << optimizer->summary();
    ss << "============================================================" << endl;
    return ss.str();
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
