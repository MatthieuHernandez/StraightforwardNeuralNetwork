#include <functional>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <boost/serialization/export.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "StraightforwardNeuralNetwork.hpp"
#include "../tools/Tools.hpp"
#include "../data/DataForClassification.hpp"
#include "../data/DataForRegression.hpp"
#include "../data/DataForMultipleClassification.hpp"

using namespace std;
using namespace chrono;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(StraightforwardNeuralNetwork)

StraightforwardNeuralNetwork::~StraightforwardNeuralNetwork()
{
    this->stopTraining();
};

StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(int numberOfInputs, vector<LayerModel> models)
    : NeuralNetwork(numberOfInputs, models)
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

vector<float> StraightforwardNeuralNetwork::computeOutput(const vector<float>& inputs)
{
    return this->output(inputs);
}

int StraightforwardNeuralNetwork::computeCluster(const vector<float>& inputs)
{
    const auto outputs = this->output(inputs);
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

template <>
void StraightforwardNeuralNetwork::startTraining(Data& data)
{
    if (typeid(data) == typeid(DataForRegression))
    {
        this->startTraining(dynamic_cast<DataForRegression&>(data));
    }
    if (typeid(data) == typeid(DataForClassification))
    {
        this->startTraining(dynamic_cast<DataForClassification&>(data));
    }
    if (typeid(data) == typeid(DataForMultipleClassification))
    {
        this->startTraining(dynamic_cast<DataForMultipleClassification&>(data));
    }
}

template <typename TData>
void StraightforwardNeuralNetwork::startTraining(TData& data)
{
    log<complete>("Start training");
    if (!this->validData(data))
        throw runtime_error("Data has not the same format as the neural network");
    this->stopTraining();
    this->isIdle = false;
    log<complete>("Start a new thread");
    this->thread = std::thread(&StraightforwardNeuralNetwork::train<TData>, this, std::ref(data));
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
    while(true) 
    {
        this_thread::sleep_for(1ms);
        auto epochs =  this->getNumberOfIteration();
        auto accuracy = this->getGlobalClusteringRate();
        auto durationMs = duration_cast<std::chrono::milliseconds>(system_clock::now() - startWait).count();
        
        if(wait.isOver(epochs, accuracy, durationMs))
            break;
    }
}

template <typename TData>
void StraightforwardNeuralNetwork::train(TData& data)
{
    this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
    this->wantToStopTraining = false;

    for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
    {
        log<minimal>("Iteration: " + to_string(this->numberOfIteration));
        this->evaluate(data);
        data.shuffle();

        for (currentIndex = 0; currentIndex < this->numberOfTrainingsBetweenTwoEvaluations && !this->wantToStopTraining;
             currentIndex ++)
        {
            this->trainOnce(data.getTrainingData(currentIndex),
                            data.getTrainingOutputs(currentIndex));
        }
    }
}

template <>
void StraightforwardNeuralNetwork::evaluate(Data& data)
{
    if (typeid(data) == typeid(DataForRegression))
    {
        this->evaluate(dynamic_cast<DataForRegression&>(data));
    }
    if (typeid(data) == typeid(DataForClassification))
    {
        this->evaluate(dynamic_cast<DataForClassification&>(data));
    }
    if (typeid(data) == typeid(DataForMultipleClassification))
    {
        this->evaluate(dynamic_cast<DataForMultipleClassification&>(data));
    }
}

template <typename TData>
void StraightforwardNeuralNetwork::evaluate(TData& data)
{
    this->startTesting();
    for (currentIndex = 0; currentIndex < data.sets[testing].size; currentIndex++)
    {
        if (this->wantToStopTraining)
            return;
        this->evaluateOnce(data);
    }
    this->stopTesting();
    if (this->autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
    {
            this->saveAs(autoSaveFilePath);
    }
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(DataForRegression& data)
{
    this->evaluateOnceForRegression(
                data.getTestingData(this->currentIndex),
                data.getTestingOutputs(this->currentIndex), data.getValue());
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(DataForClassification& data)
{
    this->evaluateOnceForClassification(
                data.getTestingData(this->currentIndex),
                data.getTestingLabel(this->currentIndex));
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(DataForMultipleClassification& data)
{
    this->evaluateOnceForMultipleClassification(
        data.getTestingData(this->currentIndex),
        data.getTestingOutputs(this->currentIndex), data.getValue());
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
    StraightforwardNeuralNetwork* neuralNetwork;
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
