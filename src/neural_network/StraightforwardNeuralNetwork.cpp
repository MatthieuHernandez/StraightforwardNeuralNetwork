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
using namespace chrono;
using namespace snn;
using namespace snn::internal;

BOOST_CLASS_EXPORT(StraightforwardNeuralNetwork)

StraightforwardNeuralNetwork::~StraightforwardNeuralNetwork()
{
    this->stopTraining();
};

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(vector<LayerModel> models)
    : NeuralNetwork(models)
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
    if(!this->isIdle)
        throw std::runtime_error("StraightforwardNeuralNetwork must be idle to be copy");
    this->autoSaveFilePath = neuralNetwork.autoSaveFilePath;
    this->autoSaveWhenBetter = neuralNetwork.autoSaveWhenBetter;
    this->currentIndex = neuralNetwork.currentIndex;
    this->numberOfIteration = neuralNetwork.numberOfIteration;
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
    for (int i = 0; i < outputs.size(); i++)
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

void StraightforwardNeuralNetwork::startTraining(Data& data)
{
    log<complete>("Start training");
    if (!this->validData(data))
        throw std::runtime_error("Data has not the same format as the neural network");
    this->stopTraining();
    this->isIdle = false;
    log<complete>("Start a new thread");
    this->thread = std::thread(&StraightforwardNeuralNetwork::train, this, std::ref(data));
}

void StraightforwardNeuralNetwork::train(Data& data)
{
    this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
    this->wantToStopTraining = false;

    this->evaluate(data);

    for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
    {
        log<minimal>("Iteration: " + std::to_string(this->numberOfIteration));
        
        data.shuffle();

        for (this->currentIndex = 0; currentIndex < this->numberOfTrainingsBetweenTwoEvaluations && !this->wantToStopTraining;
            this->currentIndex ++)
        {
            if(data.needToLearnOnTrainingData(this->currentIndex))
                this->trainOnce(data.getTrainingData(this->currentIndex),
                                data.getTrainingOutputs(this->currentIndex), data.isFirstTrainingDataOfTemporalSequence(this->currentIndex));
            else
                this->output(data.getTrainingData(this->currentIndex), data.isFirstTrainingDataOfTemporalSequence(this->currentIndex));
        }
        this->evaluate(data);
    }
}

void StraightforwardNeuralNetwork::evaluate(Data& data)
{
    this->startTesting();
    for (this->currentIndex = 0; this->currentIndex < data.sets[testing].size; this->currentIndex++)
    {
        if (this->wantToStopTraining)
            return;
        if(data.needToEvaluateOnTestingData(this->currentIndex))
            this->evaluateOnce(data);
        else
            this->output(data.getTestingData(this->currentIndex), data.isFirstTestingDataOfTemporalSequence(this->currentIndex));
    }
    this->stopTesting();
    if (this->autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
    {
        this->saveAs(autoSaveFilePath);
    }
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(Data& data)
{
    switch (data.typeOfProblem)
    {
    case classification:
        this->evaluateOnceForClassification(data.getTestingData(this->currentIndex),
                                            data.getTestingLabel(this->currentIndex),
                                            data.isFirstTestingDataOfTemporalSequence(this->currentIndex));
        break;
    case multipleClassification:
        this->evaluateOnceForMultipleClassification(data.getTestingData(this->currentIndex),
                                                    data.getTestingOutputs(this->currentIndex), data.getSeparator(),
                                                    data.isFirstTestingDataOfTemporalSequence(this->currentIndex));
        break;
    case regression:
        this->evaluateOnceForRegression(data.getTestingData(this->currentIndex),
                                        data.getTestingOutputs(this->currentIndex),
                                        data.getPrecision(),
                                        data.isFirstTestingDataOfTemporalSequence(this->currentIndex));
        break;
    default:
        throw NotImplementedException();
    }
}


void StraightforwardNeuralNetwork::stopTraining()
{
    this->wantToStopTraining = true;
    if (this->thread.joinable())
    {
        log<minimal>("Closing a thread");
        this->thread.join();
        log<complete>("Thread closed");
    }
    this->currentIndex = 0;
    this->numberOfIteration = 0;
    this->isIdle = true;
}

void StraightforwardNeuralNetwork::waitFor(Wait wait) const
{
    auto startWait = system_clock::now();
    while (true)
    {
        this_thread::sleep_for(1ms);

        const auto epochs = this->getNumberOfIteration();
        const auto accuracy = this->getGlobalClusteringRate();
        const auto mae = this->getMeanAbsoluteError();
        const auto durationMs = duration_cast<milliseconds>(system_clock::now() - startWait).count();

        if (wait.isOver(epochs, accuracy, mae, durationMs))
            break;
    }
}

int StraightforwardNeuralNetwork::isValid() const
{
    return this->NeuralNetwork::isValid();
}

bool StraightforwardNeuralNetwork::validData(const Data& data) const
{
    if(data.numberOfLabel == this->getNumberOfOutputs()
    && data.sizeOfData == this->getNumberOfInputs())
        return true;
    return false;
}


void StraightforwardNeuralNetwork::saveAs(string filePath)
{
    this->stopTraining();
    this->autoSaveFilePath = filePath;
    ofstream ofs(filePath);
    boost::archive::text_oarchive archive(ofs);
    archive << this;
}

StraightforwardNeuralNetwork& StraightforwardNeuralNetwork::loadFrom(string filePath)
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
    && this->autoSaveWhenBetter == neuralNetwork.autoSaveWhenBetter;
}

bool StraightforwardNeuralNetwork::operator!=(const StraightforwardNeuralNetwork& neuralNetwork) const
{
    return !(*this == neuralNetwork);
}
