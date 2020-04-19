#include <algorithm>
#include <stdexcept>
#include <random>
#include <vector>
#include "Data.hpp"

#include "../tools/ExtendedExpection.hpp"
#include "../tools/Tools.hpp"
using namespace std;
using namespace snn;
using namespace internal;

Data::Data(vector<vector<float>>& trainingInputs,
           vector<vector<float>>& trainingLabels,
           vector<vector<float>>& testingInputs,
           vector<vector<float>>& testingLabels,
           float value,
           temporalType type)
{
    this->initialize(trainingInputs, trainingLabels, testingInputs, testingLabels, value, type);
}

Data::Data(vector<vector<float>>& inputs,
           vector<vector<float>>& labels,
           float value,
           temporalType type)
{
    this->initialize(inputs, labels, inputs, labels, value, type);
}

void Data::initialize(vector<vector<float>>& trainingInputs,
                      vector<vector<float>>& trainingLabels,
                      vector<vector<float>>& testingInputs,
                      vector<vector<float>>& testingLabels,
                      float value,
                      temporalType type)
{
    this->value = value;
    this->type = type;
    this->sets[training].inputs.resize(1);
    this->sets[training].inputs[0] = trainingInputs;
    this->sets[training].labels = trainingLabels;
    this->sets[testing].inputs.resize(1);
    this->sets[testing].inputs[0] = testingInputs;
    this->sets[testing].labels = testingLabels;

    this->sizeOfData = static_cast<int>(trainingInputs.back().size());
    this->numberOfLabel = static_cast<int>(trainingLabels.back().size());;
    this->sets[training].size = static_cast<int>(trainingLabels.size());
    this->sets[testing].size = static_cast<int>(testingLabels.size());

    this->normalization(-1, 1);
    internal::log<minimal>("Data loaded");
}

void Data::clearData()
{
    this->sets[training].labels.clear();
    this->sets[training].inputs.clear();
    this->sets[testing].labels.clear();
    this->sets[testing].inputs.clear();
    this->sets[training].size = 0;
    this->sets[testing].size = 0;
}

void Data::normalization(const float min, const float max)
{
    try
    {
        vector2D<float>* inputsTraining = &this->sets[training].inputs[0];
        vector2D<float>* inputsTesting = &this->sets[testing].inputs[0];
        //TODO: if the first pixel of images is always black, normalization will be wrong if testing set is different
        for (int j = 0; j < this->sizeOfData; j++)
        {
            float minValueOfVector = (*inputsTraining)[0][j];
            float maxValueOfVector = (*inputsTraining)[0][j];

            for (int i = 1; i < (*inputsTraining).size(); i++)
            {
                if ((*inputsTraining)[i][j] < minValueOfVector)
                {
                    minValueOfVector = (*inputsTraining)[i][j];
                }
                else if ((*inputsTraining)[i][j] > maxValueOfVector)
                {
                    maxValueOfVector = (*inputsTraining)[i][j];
                }
            }

            const float difference = maxValueOfVector - minValueOfVector;

            for (int i = 0; i < (*inputsTraining).size(); i++)
            {
                if (difference != 0)
                    (*inputsTraining)[i][j] = ((*inputsTraining)[i][j] - minValueOfVector) / difference;
                (*inputsTraining)[i][j] = (*inputsTraining)[i][j] * (max - min) + min;
                if (isnan((*inputsTraining)[i][j]))
                    throw exception();
            }
            for (int i = 0; i < (*inputsTesting).size(); i++)
            {
                if (difference != 0)
                    (*inputsTesting)[i][j] = ((*inputsTesting)[i][j] - minValueOfVector) / difference;
                (*inputsTesting)[i][j] = (*inputsTesting)[i][j] * (max - min) + min;
                if (isnan((*inputsTesting)[i][j]))
                    throw exception();
            }
        }
    }
    catch (exception e)
    {
        throw runtime_error("Normalization of input data failed");
    }
}

void Data::shuffle()
{
    switch (this->type)
    {
    case nonTemporal:
        this->shuffleNonTemporal();
        break;
    case temporal:
        this->shuffleTemporal();
        break;
    case continuous:
        this->shuffleContinuous();
        break;
    }
}


void Data::shuffleNonTemporal()
{
    if (indexes.empty())
    {
        indexes.resize(sets[training].size);
        for (int i = 0; i < static_cast<int>(indexes.size()); i++)
            indexes[i] = i;
    }

    random_device rd;
    mt19937 g(rd());
    std::shuffle(indexes.begin(), indexes.end(), g);
}

void Data::shuffleTemporal()
{
    throw NotImplementedException("shuffleTemporal");
}

void Data::shuffleContinuous()
{
    throw NotImplementedException("shuffleContinuous");
}

void Data::unshuffle()
{
    indexes.resize(sets[training].size);
    for (int i = 0; i < static_cast<int>(indexes.size()); ++i)
        indexes[i] = i;
}

bool Data::isValid()
{
    for (auto& input : this->sets[training].inputs[0])
    {
        for (auto& value : input)
        {
            if (value < -1
                || value > 1
                || isnan(value))
            {
                return false;
            }
        }
    }
    for (auto& input : this->sets[testing].inputs[0])
    {
        for (auto& value : input)
        {
            if (isnan(value))
            {
                return false;
            }
        }
    }
    return true;
}

const vector<float>& Data::getTrainingData(const int index)
{
    return this->sets[training].inputs[0][indexes[index]];
}

const vector<float>& Data::getTestingData(const int index)
{
    return this->sets[testing].inputs[0][index];
}

const vector<float>& Data::getTrainingOutputs(const int index)
{
    return this->sets[training].labels[indexes[index]];
}

const vector<float>& Data::getData(set set, const int index)
{
    if (set == training)
        return this->getTrainingData(index);

    return this->getTestingData(index);
}

const vector<float>& Data::getOutputs(set set, const int index)
{
    if (set == training)
        return this->getTrainingOutputs(index);

    return this->getTestingOutputs(index);
}

int Data::getLabel(set set, const int index)
{
    if (set == training)
        return this->getTrainingLabel(index);

    return this->getTestingLabel(index);
}
