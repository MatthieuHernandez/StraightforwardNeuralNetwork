#include <cmath>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>
#include "Data.hpp"
#include "../tools/Tools.hpp"
#include "CompositeForClassification.hpp"
#include "CompositeForMultipleClassification.hpp"
#include "CompositeForRegression.hpp"
#include "CompositeForContinuousData.hpp"
#include "CompositeForNonTemporalData.hpp"
#include "CompositeForTemporalData.hpp"
#include "../tools/Tools.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Data::Data(problemType typeOfProblem,
           vector<vector<float>>& trainingInputs,
           vector<vector<float>>& trainingLabels,
           vector<vector<float>>& testingInputs,
           vector<vector<float>>& testingLabels,
           temporalType typeOfTemporal,
           int numberOfRecurrence)
    : typeOfProblem(typeOfProblem), typeOfTemporal(typeOfTemporal)
{
    this->initialize(typeOfProblem,
                     trainingInputs,
                     trainingLabels,
                     testingInputs,
                     testingLabels,
                     typeOfTemporal,
                     numberOfRecurrence);
}

Data::Data(problemType typeOfProblem,
           vector<vector<float>>& inputs,
           vector<vector<float>>& labels,
           temporalType typeOfTemporal,
           int numberOfRecurrence)
    : typeOfProblem(typeOfProblem), typeOfTemporal(typeOfTemporal)
{
    this->initialize(typeOfProblem,
                     inputs,
                     labels,
                     inputs,
                     labels,
                     typeOfTemporal,
                     numberOfRecurrence);
}

Data::Data(problemType typeOfProblem,
           vector<vector<vector<float>>>& trainingInputs,
           vector<vector<float>>& trainingLabels,
           vector<vector<vector<float>>>& testingInputs,
           vector<vector<float>>& testingLabels,
           temporalType typeOfTemporal,
           int numberOfRecurrence)
    : typeOfProblem(typeOfProblem), typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != temporal)
        throw runtime_error("Vector 3D type inputs are only for temporal data.");

    this->flatten(training, trainingInputs);
    this->flatten(testing, testingInputs);

    this->initialize(typeOfProblem,
                     this->sets[training].inputs,
                     trainingLabels,
                     this->sets[testing].inputs,
                     testingLabels,
                     typeOfTemporal,
                     numberOfRecurrence);
}

Data::Data(problemType typeOfProblem,
           vector<vector<vector<float>>>& inputs,
           vector<vector<float>>& labels,
           temporalType typeOfTemporal,
           int numberOfRecurrence)
    : typeOfProblem(typeOfProblem), typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != temporal)
        throw runtime_error("Vector 3D type inputs are only for temporal data.");

    this->flatten(training, inputs);

    this->initialize(typeOfProblem,
                     this->sets[training].inputs,
                     labels,
                     this->sets[testing].inputs,
                     labels,
                     typeOfTemporal,
                     numberOfRecurrence);
}

void Data::initialize(problemType typeOfProblem,
                      vector<vector<float>>& trainingInputs,
                      vector<vector<float>>& trainingLabels,
                      vector<vector<float>>& testingInputs,
                      vector<vector<float>>& testingLabels,
                      temporalType typeOfTemporal,
                      int numberOfRecurrence)
{
    this->precision = 0.1f;
    this->separator = 0.5f;

    switch (this->typeOfProblem)
    {
    case classification:
        this->problemComposite = make_unique<CompositeForClassification>(this->sets);
        break;
    case multipleClassification:
        this->problemComposite = make_unique<CompositeForMultipleClassification>(this->sets);
        break;
    case regression:
        this->problemComposite = make_unique<CompositeForRegression>(this->sets);
        break;
    default:
        throw NotImplementedException();
    }

    switch (this->typeOfTemporal)
    {
    case nonTemporal:
        this->temporalComposite = make_unique<CompositeForNonTemporalData>(this->sets);
        break;
    case temporal:
        this->temporalComposite = make_unique<CompositeForTemporalData>(this->sets);
        break;
    case continuous:
        this->temporalComposite = make_unique<CompositeForContinuousData>(this->sets, this->numberOfRecurrence);
        break;
    default:
        throw NotImplementedException();
    }

    this->numberOfRecurrence = numberOfRecurrence;
    this->sets[training].inputs = trainingInputs;
    this->sets[training].labels = trainingLabels;
    this->sets[testing].inputs = testingInputs;
    this->sets[testing].labels = testingLabels;

    this->sizeOfData = static_cast<int>(trainingInputs.back().size());
    this->numberOfLabel = static_cast<int>(trainingLabels.back().size());;
    this->sets[training].size = static_cast<int>(trainingLabels.size());
    this->sets[testing].size = static_cast<int>(testingLabels.size());

    this->sets[training].indexesToShuffle.resize(this->sets[training].size);
    for (int i = 0; i < static_cast<int>(this->sets[training].indexesToShuffle.size()); i++)
        this->sets[training].indexesToShuffle[i] = i;

    this->normalization(-1, 1);
    internal::log<minimal>("Data loaded");

    int err = this->isValid();
    if (err != 0)
    {
        string message = string("Error ") + to_string(err) + ": Wrong parameter in the creation of data";
        throw runtime_error(message);
    }
}

void Data::flatten(set set, vector<vector<vector<float>>>& input3D)
{
    auto size = accumulate(input3D.begin(), input3D.end(), 0,
                                                        [](float sum, vector2D<float>& v)
                                                        {
                                                            return sum + v.size();
                                                        });
    this->sets[set].inputs.reserve(size);
    this->sets[set].areFirstDataOfTemporalSequence.resize(size, false);

    int i = 0;
    for (vector2D<float>& v : input3D)
    {
        std::move(v.begin(), v.end(), std::back_inserter(this->sets[set].inputs));

        this->sets[set].areFirstDataOfTemporalSequence[i] = true;
        i += v.size();
    }
    this->sets[set].size = this->sets[set].inputs.size();
}

void Data::normalization(const float min, const float max)
{
    try
    {
        vector2D<float>* inputsTraining = &this->sets[training].inputs;
        vector2D<float>* inputsTesting = &this->sets[testing].inputs;
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
    this->temporalComposite->shuffle();
}

void Data::unshuffle()
{
    this->temporalComposite->unshuffle();
}

int Data::isValid()
{
    for (auto& input : this->sets[training].inputs)
    {
        for (auto& value : input)
        {
            if (value < -1
                || value > 1
                || isnan(value))
            {
                return 401;
            }
        }
    }
    for (auto& input : this->sets[testing].inputs)
    {
        for (auto& value : input)
        {
            if (value < -1
                || value > 1
                || isnan(value))
            {
                return 401;
            }
        }
    }
    if (!this->sets[testing].indexesToShuffle.empty()
      && this->sets[training].indexesToShuffle.size() != this->sets[training].size)
        return 403;

    if(this->sets[training].size == this->sets[training].inputs.size()
    && this->sets[training].size == this->sets[training].labels.size()
    && this->sets[testing].size == this->sets[training].inputs.size()
    && this->sets[testing].size == this->sets[training].labels.size())
        return 405;

    int err = this->problemComposite->isValid();
    if (err != 0)
        return err;
    err = this->temporalComposite->isValid();
    if (err != 0)
        return err;
    return 0;
}

bool Data::isFirstTrainingDataOfTemporalSequence(const int index) const
{
    return this->temporalComposite->isFirstTrainingDataOfTemporalSequence(index);
}

bool Data::isFirstTestingDataOfTemporalSequence(const int index) const
{
    return this->temporalComposite->isFirstTestingDataOfTemporalSequence(index);
}

bool Data::needToLearnOnTrainingData(const int index) const
{
    return this->temporalComposite->needToLearnOnTrainingData(index);
}

bool Data::needToEvaluateOnTestingData(int index) const
{
    return this->temporalComposite->needToEvaluateOnTestingData(index);
}

const vector<float>& Data::getTrainingData(const int index) const
{
    const int i = this->sets[training].indexesToShuffle[index];
    return this->sets[training].inputs[i];
}

const vector<float>& Data::getTestingData(const int index) const
{
    return this->sets[testing].inputs[index];
}

int Data::getTrainingLabel(const int index) const
{
    return this->problemComposite->getTrainingLabel(index);
}

int Data::getTestingLabel(const int index) const
{
    return this->problemComposite->getTestingLabel(index);
}

const vector<float>& Data::getTrainingOutputs(const int index) const
{
    return this->problemComposite->getTrainingOutputs(index);
}

const vector<float>& Data::getTestingOutputs(const int index) const
{
    return this->problemComposite->getTestingOutputs(index);
}

const vector<float>& Data::getData(set set, const int index) const
{
    if (set == training)
        return this->getTrainingData(index);

    return this->getTestingData(index);
}

const vector<float>& Data::getOutputs(set set, const int index) const
{
    if (set == training)
        return this->getTrainingOutputs(index);

    return this->getTestingOutputs(index);
}

int Data::getLabel(set set, const int index) const
{
    if (set == training)
        return this->getTrainingLabel(index);

    return this->getTestingLabel(index);
}

void Data::setPrecision(const float value)
{
    if (this->typeOfProblem == regression)
        this->precision = value;
    else
        throw std::runtime_error("Precision is only for regression problems.");
}

float Data::getPrecision() const
{
    if (this->typeOfProblem == regression)
        return this->precision;
    else
        throw std::runtime_error("Precision is only for regression problems.");
}

void Data::setSeparator(const float value)
{
    if (this->typeOfProblem == classification
        || this->typeOfProblem == multipleClassification)
        this->separator = value;
    else
        throw std::runtime_error("Separator is only for classification and multiple classification problems.");
}

float Data::getSeparator() const
{
    if (this->typeOfProblem == classification
        || this->typeOfProblem == multipleClassification)
        return this->separator;
    else
        throw std::runtime_error("Separator is only for classification and multiple classification problems.");
}
