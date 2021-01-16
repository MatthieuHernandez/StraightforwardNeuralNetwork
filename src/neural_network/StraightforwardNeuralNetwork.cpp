#include <fstream>
#include <thread>
#include <stdexcept>
#include <boost/serialization/export.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "StraightforwardNeuralNetwork.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(StraightforwardNeuralNetwork)

StraightforwardNeuralNetwork::~StraightforwardNeuralNetwork()
{
    this->stopTrainingAsync();
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(vector<LayerModel> architecture, NeuralNetworkOptimizerModel optimizer)
    : NeuralNetwork(architecture, optimizer)
{
    int err = this->isValid();
    if (err != 0)
    {
        string message = string("Error ") + to_string(err) + ": Wrong parameter in the creation of neural networks";
        throw runtime_error(message);
    }
}

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(const StraightforwardNeuralNetwork& neuralNetwork)
    : NeuralNetwork(neuralNetwork)
{
    if (!this->isIdle)
        throw std::runtime_error("StraightforwardNeuralNetwork must be idle to be copy");
    this->autoSaveFilePath = neuralNetwork.autoSaveFilePath;
    this->autoSaveWhenBetter = neuralNetwork.autoSaveWhenBetter;
    this->index = neuralNetwork.index;
    this->epoch = neuralNetwork.epoch;
    this->numberOfTrainingsBetweenTwoEvaluations = neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
}

vector<float> StraightforwardNeuralNetwork::computeOutput(const vector<float>& inputs, bool temporalReset)
{
    return this->output(inputs, temporalReset);
}

int StraightforwardNeuralNetwork::computeCluster(const vector<float>& inputs, bool temporalReset)
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

bool StraightforwardNeuralNetwork::isTraining() const
{
    return !this->isIdle;
}

void StraightforwardNeuralNetwork::startTrainingAsync(Data& data, int batchSize, int evaluationFrequency)
{
    this->validData(data, batchSize);
    this->stopTrainingAsync();
    log<complete>("Start a new thread");
    this->thread = std::thread(&StraightforwardNeuralNetwork::trainSync, this, std::ref(data), Wait(), batchSize, evaluationFrequency);
}

void StraightforwardNeuralNetwork::train(Data& data, Wait wait, int batchSize, int evaluationFrequency)
{
    this->validData(data, batchSize);
    if (!this->isIdle)
        throw std::runtime_error("Neural network is already training");
    this->stopTrainingAsync();
    wait.startClock();
    this->trainSync(data, wait, batchSize, evaluationFrequency);
}

void StraightforwardNeuralNetwork::trainSync(Data& data, Wait wait, const int batchSize, const int evaluationFrequency)
{
    log<minimal>("Start training");
    this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
    this->wantToStopTraining = false;
    this->isIdle = false;

    if (evaluationFrequency > 0)
        this->evaluate(data);

    for (this->epoch = 0; this->continueTraining(wait); this->epoch++)
    {
        log<minimal>("Epoch: " + std::to_string(this->epoch));
        data.shuffle();
        for (this->index = 0; index + batchSize <= this->numberOfTrainingsBetweenTwoEvaluations && this->continueTraining(wait);
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
                                data.getTrainingOutputs(this->index, batchSize), data.isFirstTrainingDataOfTemporalSequence(this->index));
            else
                this->output(data.getTrainingData(this->index, batchSize), data.isFirstTrainingDataOfTemporalSequence(this->index));
        }
        if (evaluationFrequency > 0 && this->epoch % evaluationFrequency == 0)
            this->evaluate(data);
    }
    this->resetTrainingValues();
    log<minimal>("Stop training");
}

void StraightforwardNeuralNetwork::evaluate(const Data& data)
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
    }
    this->stopTesting();
    log<minimal>("Accuracy: " + std::to_string(this->getGlobalClusteringRate()));
    log<minimal>("MAE: " + std::to_string(this->getMeanAbsoluteError()));
    if (this->autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
    {
        this->saveSync(autoSaveFilePath);
    }
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(const Data& data)
{
    switch (data.typeOfProblem)
    {
    case problem::classification:
        this->evaluateOnceForClassification(data.getTestingData(this->index),
                                            data.getTestingLabel(this->index),
                                            data.getSeparator(),
                                            data.isFirstTestingDataOfTemporalSequence(this->index));
        break;
    case problem::multipleClassification:
        this->evaluateOnceForMultipleClassification(data.getTestingData(this->index),
                                                    data.getTestingOutputs(this->index),
                                                    data.getSeparator(),
                                                    data.isFirstTestingDataOfTemporalSequence(this->index));
        break;
    case problem::regression:
        this->evaluateOnceForRegression(data.getTestingData(this->index),
                                        data.getTestingOutputs(this->index),
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
    this->epoch = 0;
    this->wantToStopTraining = false;
    this->isIdle = true;
}

inline
bool StraightforwardNeuralNetwork::continueTraining(Wait wait) const
{
    const auto epochs = this->getCurrentEpoch();
    const auto accuracy = this->getGlobalClusteringRate();
    const auto mae = this->getMeanAbsoluteError();

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

        if (wait.isOver(epochs, accuracy, mae))
            break;
    }
}

int StraightforwardNeuralNetwork::isValid() const
{
    return this->NeuralNetwork::isValid();
}

void StraightforwardNeuralNetwork::validData(const Data& data, int batchSize) const
{
    if (data.numberOfLabels != this->getNumberOfOutputs() || data.sizeOfData != this->getNumberOfInputs())
        throw std::runtime_error("Data has not the same format as the neural network");
    if (batchSize != 1 && data.typeOfTemporal != nature::nonTemporal)
        throw std::runtime_error("Non temporal data cannot have batch size");
    if (batchSize < 1)
        throw std::runtime_error("Wrong batch size");
    if (batchSize > data.sets[training].size)
        throw std::runtime_error("Batch size is too large");
}

void StraightforwardNeuralNetwork::saveAs(const string filePath)
{
    if(!this->isIdle)
        throw std::runtime_error("Neural network cannot be saved during training, stop training before saving or use auto save");
    saveSync(filePath);
}

void StraightforwardNeuralNetwork::saveSync(const string filePath)
{
    this->autoSaveFilePath = filePath;
    ofstream ofs(filePath);
    boost::archive::text_oarchive archive(ofs);
    archive << this;
            log<minimal>("Neural network saved: ", filePath);
}


StraightforwardNeuralNetwork& StraightforwardNeuralNetwork::loadFrom(const string filePath)
{
    StraightforwardNeuralNetwork* neuralNetwork = nullptr;
    ifstream ifs(filePath);
    boost::archive::text_iarchive archive(ifs);
    archive >> neuralNetwork;
    return *neuralNetwork;
}

bool StraightforwardNeuralNetwork::operator==(const StraightforwardNeuralNetwork& neuralNetwork) const
{
    return this->NeuralNetwork::operator==(neuralNetwork)
        && this->autoSaveFilePath == neuralNetwork.autoSaveFilePath
        && this->autoSaveWhenBetter == neuralNetwork.autoSaveWhenBetter
        && this->wantToStopTraining == neuralNetwork.wantToStopTraining
        && this->index == neuralNetwork.index
        && this->isIdle == neuralNetwork.isIdle
        && this->epoch == neuralNetwork.epoch
        && this->numberOfTrainingsBetweenTwoEvaluations == neuralNetwork.numberOfTrainingsBetweenTwoEvaluations;
}

bool StraightforwardNeuralNetwork::operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const
{
    return !(*this == neuralNetwork);
}
