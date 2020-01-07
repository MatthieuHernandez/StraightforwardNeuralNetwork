#include <functional>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <boost/serialization/export.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "StraightforwardNeuralNetwork.hpp"
#include "../data/DataForClassification.hpp"
#include "../data/DataForRegression.hpp"
#include "../data/DataForMultipleClassification.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(StraightforwardNeuralNetwork)

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

void StraightforwardNeuralNetwork::trainingStart(Data& data)
{
    this->trainingStop();
    this->isIdle = false;
    this->thread = std::thread(&StraightforwardNeuralNetwork::train, this, std::ref(data));
    this->thread.detach();
}

void StraightforwardNeuralNetwork::train(Data& data)
{
    this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
    this->wantToStopTraining = false;

    for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
    {
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

void StraightforwardNeuralNetwork::evaluate(Data& data)
{
    const auto evaluation = selectEvaluationFunction(data);

    this->startTesting();
    for (currentIndex = 0; currentIndex < data.sets[testing].size; currentIndex++)
    {
        if (this->wantToStopTraining)
        {
            this->stopTesting();
            return;
        }

        std::invoke(evaluation, this, data);
    }
    this->stopTesting();
    if (this->autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
    {
            this->saveAs(autoSaveFilePath);
    }
}

inline
StraightforwardNeuralNetwork::evaluationFunctionPtr StraightforwardNeuralNetwork::selectEvaluationFunction(Data& data)
{
    if(typeid(data) == typeid(DataForRegression))
    {
        return &StraightforwardNeuralNetwork::evaluateOnceForRegression;
    }
    if(typeid(data) == typeid(DataForMultipleClassification))
    {
        this->separator = data.getValue();
        return &StraightforwardNeuralNetwork::evaluateOnceForMultipleClassification;
    }
    if(typeid(data) == typeid(DataForClassification))
    {
        return &StraightforwardNeuralNetwork::evaluateOnceForClassification;
    }

    throw runtime_error("wrong Data typeid");
}

inline
void StraightforwardNeuralNetwork::evaluateOnceForRegression(Data& data)
{
    this->NeuralNetwork::evaluateOnceForRegression(
                data.getTestingData(this->currentIndex),
                data.getTestingOutputs(this->currentIndex), data.getValue());
}

inline
void StraightforwardNeuralNetwork::evaluateOnceForMultipleClassification(Data& data)
{
    this->NeuralNetwork::evaluateOnceForMultipleClassification(
                data.getTestingData(this->currentIndex),
                data.getTestingOutputs(this->currentIndex), data.getValue());
}

inline
void StraightforwardNeuralNetwork::evaluateOnceForClassification(Data& data)
{
    this->NeuralNetwork::evaluateOnceForClassification(
                data.getTestingData(this->currentIndex),
                data.getTestingLabel(this->currentIndex));
}


void StraightforwardNeuralNetwork::trainingStop()
{
    this->wantToStopTraining = true;
    if (this->thread.joinable())
        this->thread.join();
    this->currentIndex = 0;
    this->numberOfIteration = 0;
    this->isIdle = true;
}

int StraightforwardNeuralNetwork::isValid() const
{
    return this->NeuralNetwork::isValid();
}


void StraightforwardNeuralNetwork::saveAs(string filePath)
{
    this->autoSaveFilePath = filePath;
    ofstream ofs(filePath);
    boost::archive::text_oarchive archive(ofs);
    archive << this;
}

StraightforwardNeuralNetwork StraightforwardNeuralNetwork::loadFrom(string filePath)
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
