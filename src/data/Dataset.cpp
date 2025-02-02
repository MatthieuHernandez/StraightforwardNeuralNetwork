#include "Dataset.hpp"

#include <algorithm>
#include <memory>
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

namespace snn
{
Dataset::Dataset(problem typeOfProblem, std::vector<std::vector<float>>& trainingInputs,
                 std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<float>>& testingInputs,
                 std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    this->initialize(trainingInputs, trainingLabels, testingInputs, testingLabels);
}

Dataset::Dataset(problem typeOfProblem, std::vector<std::vector<float>>& inputs,
                 std::vector<std::vector<float>>& labels, nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    this->initialize(inputs, labels, inputs, labels);
}

Dataset::Dataset(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& trainingInputs,
                 std::vector<std::vector<float>>& trainingLabels,
                 std::vector<std::vector<std::vector<float>>>& testingInputs,
                 std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != nature::sequential)
    {
        throw std::runtime_error("std::vector 3D type inputs are only for sequential data.");
    }

    this->flatten(this->data.training, trainingInputs);
    this->flatten(this->data.testing, testingInputs);

    this->initialize(this->data.training.inputs, trainingLabels, this->data.testing.inputs, testingLabels);
}

Dataset::Dataset(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& inputs,
                 std::vector<std::vector<float>>& labels, nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != nature::sequential)
    {
        throw std::runtime_error("std::vector 3D type inputs are only for sequential data.");
    }

    this->flatten(inputs);

    this->initialize(this->data.training.inputs, labels, this->data.testing.inputs, labels);
}

void Dataset::initialize(std::vector<std::vector<float>>& trainingInputs,
                         std::vector<std::vector<float>>& trainingLabels,
                         std::vector<std::vector<float>>& testingInputs, std::vector<std::vector<float>>& testingLabels)
{
    this->data.training.inputs = trainingInputs;
    this->data.training.labels = trainingLabels;
    this->data.testing.inputs = testingInputs;
    this->data.testing.labels = testingLabels;

    this->sizeOfData = static_cast<int>(trainingInputs.back().size());
    this->numberOfLabels = static_cast<int>(trainingLabels.back().size());
    this->data.training.size = static_cast<int>(trainingLabels.size());
    this->data.testing.size = static_cast<int>(testingLabels.size());

    batchedData.resize(this->sizeOfData, 0.0F);

    this->data.training.shuffledIndexes.resize(this->data.training.size);
    for (int i = 0; i < static_cast<int>(this->data.training.shuffledIndexes.size()); i++)
    {
        this->data.training.shuffledIndexes[i] = i;
    }

    switch (this->typeOfProblem)
    {
        case problem::classification:
            this->problemComposite =
                std::make_unique<internal::CompositeForClassification>(&this->data, this->numberOfLabels);
            break;
        case problem::multipleClassification:
            this->problemComposite =
                std::make_unique<internal::CompositeForMultipleClassification>(&this->data, this->numberOfLabels);
            break;
        case problem::regression:
            this->problemComposite =
                std::make_unique<internal::CompositeForRegression>(&this->data, this->numberOfLabels);
            break;
        default:
            throw NotImplementedException();
    }

    switch (this->typeOfTemporal)
    {
        case nature::nonTemporal:
            this->temporalComposite = std::make_unique<internal::CompositeForNonTemporalData>(&this->data);
            break;
        case nature::sequential:
            this->temporalComposite = std::make_unique<internal::CompositeForTemporalData>(&this->data);
            break;
        case nature::timeSeries:
            this->temporalComposite =
                std::make_unique<internal::CompositeForTimeSeries>(&this->data, this->numberOfRecurrences);
            break;
        default:
            throw NotImplementedException();
    }

    const auto err = this->isValid();
    if (err != errorType::noError)
    {
        const auto message = std::string("Error ") + tools::toString(err) + ": Wrong parameter in the creation of data";
        throw std::runtime_error(message);
    }

    tools::log<minimal>("Data loaded");
}

void Dataset::flatten(internal::Set& set, std::vector<std::vector<std::vector<float>>>& input3D)
{
    set.numberOfTemporalSequence = static_cast<int>(input3D.size());
    const size_t size = accumulate(input3D.begin(), input3D.end(), static_cast<size_t>(0),
                                   [](size_t sum, vector2D<float>& val) { return sum + val.size(); });
    set.inputs.reserve(size);
    set.areFirstDataOfTemporalSequence.resize(size, false);
    if (set.type == setType::testing)
    {
        set.needToEvaluateOnData.resize(size, false);
    }

    size_t count = 0;
    for (auto& input : input3D)
    {
        std::ranges::move(input, back_inserter(set.inputs));

        set.areFirstDataOfTemporalSequence[count] = true;
        count += input.size();
        if (set.type == setType::testing)
        {
            this->data.testing.needToEvaluateOnData[count - 1] = true;
        }
    }
    set.size = static_cast<int>(set.inputs.size());
}

void Dataset::flatten(std::vector<std::vector<std::vector<float>>>& input3D)
{
    this->data.training.numberOfTemporalSequence = static_cast<int>(input3D.size());
    this->data.testing.numberOfTemporalSequence = static_cast<int>(input3D.size());
    const size_t size = accumulate(input3D.begin(), input3D.end(), static_cast<size_t>(0),
                                   [](size_t sum, vector2D<float>& val) { return sum + val.size(); });
    this->data.training.inputs.reserve(size);
    this->data.training.areFirstDataOfTemporalSequence.resize(size, false);
    this->data.testing.areFirstDataOfTemporalSequence.resize(size, false);
    this->data.testing.needToEvaluateOnData.resize(size, false);

    size_t count = 0;
    for (vector2D<float>& input : input3D)
    {
        std::ranges::move(input, back_inserter(this->data.training.inputs));

        this->data.training.areFirstDataOfTemporalSequence[count] = true;
        this->data.testing.areFirstDataOfTemporalSequence[count] = true;
        count += input.size();
        this->data.testing.needToEvaluateOnData[count - 1] = true;
    }
    this->data.testing.inputs = this->data.training.inputs;
    this->data.training.size = static_cast<int>(this->data.training.inputs.size());
    this->data.testing.size = static_cast<int>(this->data.testing.inputs.size());
}

void Dataset::normalize(const float min, const float max)
{
    try
    {
        vector2D<float>& inputsTraining = this->data.training.inputs;
        vector2D<float>& inputsTesting = this->data.testing.inputs;
        // TODO(matth): if the first pixel of images is always black, normalization will be wrong if testing set is
        // different
        for (int j = 0; j < this->sizeOfData; ++j)
        {
            float minValueOfvector = inputsTraining[0][j];
            float maxValueOfvector = inputsTraining[0][j];

            for (size_t i = 1; i < inputsTraining.size(); ++i)
            {
                if (inputsTraining[i][j] < minValueOfvector)
                {
                    minValueOfvector = inputsTraining[i][j];
                }
                else if (inputsTraining[i][j] > maxValueOfvector)
                {
                    maxValueOfvector = inputsTraining[i][j];
                }
            }

            const float difference = maxValueOfvector - minValueOfvector;

            for (auto& input : inputsTraining)
            {
                if (difference != 0)
                {
                    input[j] = (input[j] - minValueOfvector) / difference;
                }
                input[j] = input[j] * (max - min) + min;
            }
            for (auto& input : inputsTesting)
            {
                if (difference != 0)
                {
                    input[j] = (input[j] - minValueOfvector) / difference;
                }
                input[j] = input[j] * (max - min) + min;
            }
        }
    }
    catch (std::exception&)
    {
        throw std::runtime_error("Normalization of input data failed");
    }
}

void Dataset::shuffle() { this->temporalComposite->shuffle(); }

void Dataset::unshuffle() { this->temporalComposite->unshuffle(); }

auto Dataset::isValid() const -> errorType
{
    if (!this->data.testing.shuffledIndexes.empty() &&
        this->data.training.size != static_cast<int>(this->data.training.shuffledIndexes.size()))
    {
        return errorType::dataWrongIdexes;
    }

    if (this->data.training.size != static_cast<int>(this->data.training.inputs.size()) &&
        this->data.training.size != static_cast<int>(this->data.training.labels.size()) &&
        this->data.testing.size != static_cast<int>(this->data.training.inputs.size()) &&
        this->data.testing.size != static_cast<int>(this->data.training.labels.size()))
    {
        return errorType::dataWrongSize;
    }

    auto err = this->problemComposite->isValid();
    if (err != errorType::noError)
    {
        return err;
    }
    err = this->temporalComposite->isValid();
    if (err != errorType::noError)
    {
        return err;
    }
    return errorType::noError;
}

auto Dataset::isFirstTrainingDataOfTemporalSequence(const int index) const -> bool
{
    return this->temporalComposite->isFirstTrainingDataOfTemporalSequence(index);
}

auto Dataset::isFirstTestingDataOfTemporalSequence(const int index) const -> bool
{
    return this->temporalComposite->isFirstTestingDataOfTemporalSequence(index);
}

auto Dataset::needToLearnOnTrainingData(const int index) const -> bool
{
    return this->temporalComposite->needToTrainOnTrainingData(index);
}

auto Dataset::needToEvaluateOnTestingData(int index) const -> bool
{
    return this->temporalComposite->needToEvaluateOnTestingData(index);
}

auto Dataset::getTrainingData(const int index, const int batchSize) -> const std::vector<float>&
{
    auto idx = this->data.training.shuffledIndexes[index];
    if (batchSize <= 1)
    {
        return this->data.training.inputs[idx];
    }
    idx = this->data.training.shuffledIndexes[index];
    const auto firstData = this->data.training.inputs[idx];

    std::ranges::copy(firstData, batchedData.begin());
    const auto end = index + batchSize;
    for (int j = index + 1; j < end; ++j)
    {
        idx = this->data.training.shuffledIndexes[j];
        const auto dataToAdd = this->data.training.inputs[idx];
        std::ranges::transform(batchedData, dataToAdd, batchedData.begin(),
                               [](float total, float data) { return total + data; });
    }
    std::ranges::transform(batchedData, batchedData.begin(),
                           [&batchSize](float total) { return total / static_cast<float>(batchSize); });
    return batchedData;
}

auto Dataset::getTestingData(const int index) const -> const std::vector<float>&
{
    return this->data.testing.inputs[index];
}

auto Dataset::getTrainingLabel(const int index) const -> int { return this->problemComposite->getTrainingLabel(index); }

auto Dataset::getTestingLabel(const int index) const -> int { return this->problemComposite->getTestingLabel(index); }

auto Dataset::getTrainingOutputs(const int index, const int batchSize) -> const std::vector<float>&
{
    return this->problemComposite->getTrainingOutputs(index, batchSize);
}

auto Dataset::getTestingOutputs(const int index) const -> const std::vector<float>&
{
    return this->problemComposite->getTestingOutputs(index);
}

void Dataset::setPrecision(const float value)
{
    if (this->typeOfProblem == problem::regression)
    {
        this->precision = value;
    }
    else
    {
        throw std::runtime_error("Precision is only for regression problems.");
    }
}

auto Dataset::getPrecision() const -> float
{
    if (this->typeOfProblem == problem::regression)
    {
        return this->precision;
    }
    throw std::runtime_error("Precision is only for regression problems.");
}

void Dataset::setSeparator(const float value)
{
    if (this->typeOfProblem == problem::classification || this->typeOfProblem == problem::multipleClassification)
    {
        this->separator = value;
    }
    else
    {
        throw std::runtime_error("Separator is only for classification and multiple classification problems.");
    }
}

auto Dataset::getSeparator() const -> float
{
    if (this->typeOfProblem == problem::classification || this->typeOfProblem == problem::multipleClassification)
    {
        return this->separator;
    }
    throw std::runtime_error("Separator is only for classification and multiple classification problems.");
}
}  // namespace snn
