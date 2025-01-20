#include "Data.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "CompositeForClassification.hpp"
#include "CompositeForMultipleClassification.hpp"
#include "CompositeForNonTemporalData.hpp"
#include "CompositeForRegression.hpp"
#include "CompositeForTemporalData.hpp"
#include "CompositeForTimeSeries.hpp"
#include "Error.hpp"
#include "ExtendedExpection.hpp"
#include "Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Data::Data(problem typeOfProblem, vector<vector<float>>& trainingInputs, vector<vector<float>>& trainingLabels,
           vector<vector<float>>& testingInputs, vector<vector<float>>& testingLabels, nature typeOfTemporal,
           int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    this->initialize(trainingInputs, trainingLabels, testingInputs, testingLabels);
}

Data::Data(problem typeOfProblem, vector<vector<float>>& inputs, vector<vector<float>>& labels, nature typeOfTemporal,
           int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    this->initialize(inputs, labels, inputs, labels);
}

Data::Data(problem typeOfProblem, vector<vector<vector<float>>>& trainingInputs, vector<vector<float>>& trainingLabels,
           vector<vector<vector<float>>>& testingInputs, vector<vector<float>>& testingLabels, nature typeOfTemporal,
           int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != nature::sequential)
        throw runtime_error("Vector 3D type inputs are only for sequential data.");

    this->flatten(training, trainingInputs);
    this->flatten(testing, testingInputs);

    this->initialize(this->sets[training].inputs, trainingLabels, this->sets[testing].inputs, testingLabels);
}

Data::Data(problem typeOfProblem, vector<vector<vector<float>>>& inputs, vector<vector<float>>& labels,
           nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != nature::sequential)
        throw runtime_error("Vector 3D type inputs are only for sequential data.");

    this->flatten(inputs);

    this->initialize(this->sets[training].inputs, labels, this->sets[testing].inputs, labels);
}

void Data::initialize(vector<vector<float>>& trainingInputs, vector<vector<float>>& trainingLabels,
                      vector<vector<float>>& testingInputs, vector<vector<float>>& testingLabels)
{
    this->precision = 0.1f;
    this->separator = 0.5f;
    this->sets[training].inputs = trainingInputs;
    this->sets[training].labels = trainingLabels;
    this->sets[testing].inputs = testingInputs;
    this->sets[testing].labels = testingLabels;

    this->sizeOfData = static_cast<int>(trainingInputs.back().size());
    this->numberOfLabels = static_cast<int>(trainingLabels.back().size());
    this->sets[training].size = static_cast<int>(trainingLabels.size());
    this->sets[testing].size = static_cast<int>(testingLabels.size());

    this->sets[training].shuffledIndexes.resize(this->sets[training].size);
    for (int i = 0; i < static_cast<int>(this->sets[training].shuffledIndexes.size()); i++)
        this->sets[training].shuffledIndexes[i] = i;

    switch (this->typeOfProblem)
    {
        case problem::classification:
            this->problemComposite = make_unique<CompositeForClassification>(this->sets, this->numberOfLabels);
            break;
        case problem::multipleClassification:
            this->problemComposite = make_unique<CompositeForMultipleClassification>(this->sets, this->numberOfLabels);
            break;
        case problem::regression:
            this->problemComposite = make_unique<CompositeForRegression>(this->sets, this->numberOfLabels);
            break;
        default:
            throw NotImplementedException();
    }

    switch (this->typeOfTemporal)
    {
        case nature::nonTemporal:
            this->temporalComposite = make_unique<CompositeForNonTemporalData>(this->sets);
            break;
        case nature::sequential:
            this->temporalComposite = make_unique<CompositeForTemporalData>(this->sets);
            break;
        case nature::timeSeries:
            this->temporalComposite = make_unique<CompositeForTimeSeries>(this->sets, this->numberOfRecurrences);
            break;
        default:
            throw NotImplementedException();
    }

    const auto err = this->isValid();
    if (err != ErrorType::noError)
    {
        const string message = string("Error ") + tools::toString(err) + ": Wrong parameter in the creation of data";
        throw runtime_error(message);
    }

    tools::log<minimal>("Data loaded");
}

void Data::flatten(set set, vector<vector<vector<float>>>& input3D)
{
    this->sets[set].numberOfTemporalSequence = (int)input3D.size();
    size_t size = accumulate(input3D.begin(), input3D.end(), (size_t)0,
                             [](size_t sum, vector2D<float>& v) { return sum + v.size(); });
    this->sets[set].inputs.reserve(size);
    this->sets[set].areFirstDataOfTemporalSequence.resize(size, false);
    if (set == testing) this->sets[set].needToEvaluateOnData.resize(size, false);

    size_t i = 0;
    for (vector2D<float>& v : input3D)
    {
        ranges::move(v, back_inserter(this->sets[set].inputs));

        this->sets[set].areFirstDataOfTemporalSequence[i] = true;
        i += v.size();
        if (set == testing) this->sets[testing].needToEvaluateOnData[i - 1] = true;
    }
    this->sets[set].size = (int)this->sets[set].inputs.size();
}

void Data::flatten(vector<vector<vector<float>>>& input3D)
{
    this->sets[training].numberOfTemporalSequence = (int)input3D.size();
    this->sets[testing].numberOfTemporalSequence = (int)input3D.size();
    size_t size = accumulate(input3D.begin(), input3D.end(), (size_t)0,
                             [](size_t sum, vector2D<float>& v) { return sum + v.size(); });
    this->sets[training].inputs.reserve(size);
    this->sets[training].areFirstDataOfTemporalSequence.resize(size, false);
    this->sets[testing].areFirstDataOfTemporalSequence.resize(size, false);
    this->sets[testing].needToEvaluateOnData.resize(size, false);

    size_t i = 0;
    for (vector2D<float>& v : input3D)
    {
        ranges::move(v, back_inserter(this->sets[training].inputs));

        this->sets[training].areFirstDataOfTemporalSequence[i] = true;
        this->sets[testing].areFirstDataOfTemporalSequence[i] = true;
        i += v.size();
        this->sets[testing].needToEvaluateOnData[i - 1] = true;
    }
    this->sets[testing].inputs = this->sets[training].inputs;
    this->sets[training].size = (int)this->sets[training].inputs.size();
    this->sets[testing].size = (int)this->sets[testing].inputs.size();
}

void Data::normalize(const float min, const float max)
{
    try
    {
        vector2D<float>& inputsTraining = this->sets[training].inputs;
        vector2D<float>& inputsTesting = this->sets[testing].inputs;
        // TODO: if the first pixel of images is always black, normalization will be wrong if testing set is different
        for (int j = 0; j < this->sizeOfData; j++)
        {
            float minValueOfVector = inputsTraining[0][j];
            float maxValueOfVector = inputsTraining[0][j];

            for (size_t i = 1; i < inputsTraining.size(); i++)
            {
                if (inputsTraining[i][j] < minValueOfVector)
                {
                    minValueOfVector = inputsTraining[i][j];
                }
                else if (inputsTraining[i][j] > maxValueOfVector)
                {
                    maxValueOfVector = inputsTraining[i][j];
                }
            }

            const float difference = maxValueOfVector - minValueOfVector;

            for (size_t i = 0; i < inputsTraining.size(); i++)
            {
                if (difference != 0) inputsTraining[i][j] = (inputsTraining[i][j] - minValueOfVector) / difference;
                inputsTraining[i][j] = inputsTraining[i][j] * (max - min) + min;
            }
            for (size_t i = 0; i < inputsTesting.size(); i++)
            {
                if (difference != 0) inputsTesting[i][j] = (inputsTesting[i][j] - minValueOfVector) / difference;
                inputsTesting[i][j] = inputsTesting[i][j] * (max - min) + min;
            }
        }
    }
    catch (exception&)
    {
        throw runtime_error("Normalization of input data failed");
    }
}

void Data::shuffle() { this->temporalComposite->shuffle(); }

void Data::unshuffle() { this->temporalComposite->unshuffle(); }

auto Data::isValid() const -> ErrorType
{
    if (!this->sets[testing].shuffledIndexes.empty() &&
        this->sets[training].size != (int)this->sets[training].shuffledIndexes.size())
    {
        return ErrorType::dataWrongIdexes;
    }

    if (this->sets[training].size != (int)this->sets[training].inputs.size() &&
        this->sets[training].size != (int)this->sets[training].labels.size() &&
        this->sets[testing].size != (int)this->sets[training].inputs.size() &&
        this->sets[testing].size != (int)this->sets[training].labels.size())
    {
        return ErrorType::dataWrongSize;
    }

    auto err = this->problemComposite->isValid();
    if (err != ErrorType::noError)
    {
        return err;
    }
    err = this->temporalComposite->isValid();
    if (err != ErrorType::noError)
    {
        return err;
    }
    return ErrorType::noError;
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
    return this->temporalComposite->needToTrainOnTrainingData(index);
}

bool Data::needToEvaluateOnTestingData(int index) const
{
    return this->temporalComposite->needToEvaluateOnTestingData(index);
}

const vector<float>& Data::getTrainingData(const int index, const int batchSize)
{
    int i = this->sets[training].shuffledIndexes[index];
    if (batchSize <= 1) return this->sets[training].inputs[i];

    batchedData.resize(this->sizeOfData);

    i = this->sets[training].shuffledIndexes[index];
    const auto data0 = this->sets[training].inputs[i];
    i = this->sets[training].shuffledIndexes[index + 1];
    const auto data1 = this->sets[training].inputs[i];
    ranges::transform(data0, data1, batchedData.begin(), plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        i = this->sets[training].shuffledIndexes[j];
        const auto data = this->sets[training].inputs[i];
        ranges::transform(batchedData, data, batchedData.begin(), std::plus<float>());
    }
    ranges::transform(batchedData, batchedData.begin(),
                      bind(divides<float>(), placeholders::_1, static_cast<float>(batchSize)));
    return batchedData;
}

const vector<float>& Data::getTestingData(const int index) const { return this->sets[testing].inputs[index]; }

int Data::getTrainingLabel(const int index) const { return this->problemComposite->getTrainingLabel(index); }

int Data::getTestingLabel(const int index) const { return this->problemComposite->getTestingLabel(index); }

const vector<float>& Data::getTrainingOutputs(const int index, const int batchSize)
{
    return this->problemComposite->getTrainingOutputs(index, batchSize);
}

const vector<float>& Data::getTestingOutputs(const int index) const
{
    return this->problemComposite->getTestingOutputs(index);
}

const vector<float>& Data::getData(set set, const int index)
{
    if (set == training) return this->getTrainingData(index);

    return this->getTestingData(index);
}

const vector<float>& Data::getOutputs(set set, const int index)
{
    if (set == training) return this->getTrainingOutputs(index);

    return this->getTestingOutputs(index);
}

int Data::getLabel(set set, const int index) const
{
    if (set == training) return this->getTrainingLabel(index);

    return this->getTestingLabel(index);
}

void Data::setPrecision(const float value)
{
    if (this->typeOfProblem == problem::regression)
        this->precision = value;
    else
        throw runtime_error("Precision is only for regression problems.");
}

float Data::getPrecision() const
{
    if (this->typeOfProblem == problem::regression)
        return this->precision;
    else
        throw runtime_error("Precision is only for regression problems.");
}

void Data::setSeparator(const float value)
{
    if (this->typeOfProblem == problem::classification || this->typeOfProblem == problem::multipleClassification)
        this->separator = value;
    else
        throw runtime_error("Separator is only for classification and multiple classification problems.");
}

float Data::getSeparator() const
{
    if (this->typeOfProblem == problem::classification || this->typeOfProblem == problem::multipleClassification)
        return this->separator;
    else
        throw runtime_error("Separator is only for classification and multiple classification problems.");
}
