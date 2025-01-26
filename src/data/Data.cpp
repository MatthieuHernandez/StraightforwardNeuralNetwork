#include "Data.hpp"

#include <algorithm>
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

namespace snn
{
Data::Data(problem typeOfProblem, std::vector<std::vector<float>>& trainingInputs,
           std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<float>>& testingInputs,
           std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    this->initialize(trainingInputs, trainingLabels, testingInputs, testingLabels);
}

Data::Data(problem typeOfProblem, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels,
           nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    this->initialize(inputs, labels, inputs, labels);
}

Data::Data(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& trainingInputs,
           std::vector<std::vector<float>>& trainingLabels, std::vector<std::vector<std::vector<float>>>& testingInputs,
           std::vector<std::vector<float>>& testingLabels, nature typeOfTemporal, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences),
      typeOfProblem(typeOfProblem),
      typeOfTemporal(typeOfTemporal)
{
    if (this->typeOfTemporal != nature::sequential)
    {
        throw std::runtime_error("std::vector 3D type inputs are only for sequential data.");
    }

    this->flatten(this->set.training, trainingInputs);
    this->flatten(this->set.testing, testingInputs);

    this->initialize(this->set.training.inputs, trainingLabels, this->set.testing.inputs, testingLabels);
}

Data::Data(problem typeOfProblem, std::vector<std::vector<std::vector<float>>>& inputs,
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

    this->initialize(this->set.training.inputs, labels, this->set.testing.inputs, labels);
}

void Data::initialize(std::vector<std::vector<float>>& trainingInputs, std::vector<std::vector<float>>& trainingLabels,
                      std::vector<std::vector<float>>& testingInputs, std::vector<std::vector<float>>& testingLabels)
{
    this->precision = 0.1F;
    this->separator = 0.5F;
    this->set.training.inputs = trainingInputs;
    this->set.training.labels = trainingLabels;
    this->set.testing.inputs = testingInputs;
    this->set.testing.labels = testingLabels;

    this->sizeOfData = static_cast<int>(trainingInputs.back().size());
    this->numberOfLabels = static_cast<int>(trainingLabels.back().size());
    this->set.training.size = static_cast<int>(trainingLabels.size());
    this->set.testing.size = static_cast<int>(testingLabels.size());

    this->set.training.shuffledIndexes.resize(this->set.training.size);
    for (int i = 0; i < static_cast<int>(this->set.training.shuffledIndexes.size()); i++)
    {
        this->set.training.shuffledIndexes[i] = i;
    }

    switch (this->typeOfProblem)
    {
        case problem::classification:
            this->problemComposite =
                std::make_unique<internal::CompositeForClassification>(&this->set, this->numberOfLabels);
            break;
        case problem::multipleClassification:
            this->problemComposite =
                std::make_unique<internal::CompositeForMultipleClassification>(&this->set, this->numberOfLabels);
            break;
        case problem::regression:
            this->problemComposite =
                std::make_unique<internal::CompositeForRegression>(&this->set, this->numberOfLabels);
            break;
        default:
            throw NotImplementedException();
    }

    switch (this->typeOfTemporal)
    {
        case nature::nonTemporal:
            this->temporalComposite = std::make_unique<internal::CompositeForNonTemporalData>(&this->set);
            break;
        case nature::sequential:
            this->temporalComposite = std::make_unique<internal::CompositeForTemporalData>(&this->set);
            break;
        case nature::timeSeries:
            this->temporalComposite =
                std::make_unique<internal::CompositeForTimeSeries>(&this->set, this->numberOfRecurrences);
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

void Data::flatten(Set& set, std::vector<std::vector<std::vector<float>>>& input3D)
{
    set.numberOfTemporalSequence = static_cast<int>(input3D.size());
    size_t size = accumulate(input3D.begin(), input3D.end(), static_cast<size_t>(0),
                             [](size_t sum, vector2D<float>& v) { return sum + v.size(); });
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
            this->set.testing.needToEvaluateOnData[count - 1] = true;
        }
    }
    set.size = static_cast<int>(set.inputs.size());
}

void Data::flatten(std::vector<std::vector<std::vector<float>>>& input3D)
{
    this->set.training.numberOfTemporalSequence = static_cast<int>(input3D.size());
    this->set.testing.numberOfTemporalSequence = static_cast<int>(input3D.size());
    size_t size = accumulate(input3D.begin(), input3D.end(), static_cast<size_t>(0),
                             [](size_t sum, vector2D<float>& v) { return sum + v.size(); });
    this->set.training.inputs.reserve(size);
    this->set.training.areFirstDataOfTemporalSequence.resize(size, false);
    this->set.testing.areFirstDataOfTemporalSequence.resize(size, false);
    this->set.testing.needToEvaluateOnData.resize(size, false);

    size_t count = 0;
    for (vector2D<float>& input : input3D)
    {
        std::ranges::move(input, back_inserter(this->set.training.inputs));

        this->set.training.areFirstDataOfTemporalSequence[count] = true;
        this->set.testing.areFirstDataOfTemporalSequence[count] = true;
        count += input.size();
        this->set.testing.needToEvaluateOnData[count - 1] = true;
    }
    this->set.testing.inputs = this->set.training.inputs;
    this->set.training.size = static_cast<int>(this->set.training.inputs.size());
    this->set.testing.size = static_cast<int>(this->set.testing.inputs.size());
}

void Data::normalize(const float min, const float max)
{
    try
    {
        vector2D<float>& inputsTraining = this->set.training.inputs;
        vector2D<float>& inputsTesting = this->set.testing.inputs;
        // TODO(matth): if the first pixel of images is always black, normalization will be wrong if testing set is
        // different
        for (int j = 0; j < this->sizeOfData; j++)
        {
            float minValueOfvector = inputsTraining[0][j];
            float maxValueOfvector = inputsTraining[0][j];

            for (size_t i = 1; i < inputsTraining.size(); i++)
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

void Data::shuffle() { this->temporalComposite->shuffle(); }

void Data::unshuffle() { this->temporalComposite->unshuffle(); }

auto Data::isValid() const -> errorType
{
    if (!this->set.testing.shuffledIndexes.empty() &&
        this->set.training.size != this->set.training.shuffledIndexes.size())
    {
        return errorType::dataWrongIdexes;
    }

    if (this->set.training.size != this->set.training.inputs.size() &&
        this->set.training.size != this->set.training.labels.size() &&
        this->set.testing.size != this->set.training.inputs.size() &&
        this->set.testing.size != this->set.training.labels.size())
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

auto Data::isFirstTrainingDataOfTemporalSequence(const int index) const -> bool
{
    return this->temporalComposite->isFirstTrainingDataOfTemporalSequence(index);
}

auto Data::isFirstTestingDataOfTemporalSequence(const int index) const -> bool
{
    return this->temporalComposite->isFirstTestingDataOfTemporalSequence(index);
}

auto Data::needToLearnOnTrainingData(const int index) const -> bool
{
    return this->temporalComposite->needToTrainOnTrainingData(index);
}

auto Data::needToEvaluateOnTestingData(int index) const -> bool
{
    return this->temporalComposite->needToEvaluateOnTestingData(index);
}

auto Data::getTrainingData(const int index, const int batchSize) -> const std::vector<float>&
{
    auto idx = this->set.training.shuffledIndexes[index];
    if (batchSize <= 1)
    {
        return this->set.training.inputs[idx];
    }

    batchedData.resize(this->sizeOfData);

    idx = this->set.training.shuffledIndexes[index];
    const auto data0 = this->set.training.inputs[idx];
    idx = this->set.training.shuffledIndexes[index + 1];
    const auto data1 = this->set.training.inputs[idx];
    std::ranges::transform(data0, data1, batchedData.begin(), std::plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        idx = this->set.training.shuffledIndexes[j];
        const auto data = this->set.training.inputs[idx];
        std::ranges::transform(batchedData, data, batchedData.begin(), std::plus<float>());
    }
    std::ranges::transform(batchedData, batchedData.begin(),
                           bind(std::divides<float>(), std::placeholders::_1, static_cast<float>(batchSize)));
    return batchedData;
}

auto Data::getTestingData(const int index) const -> const std::vector<float>&
{
    return this->set.testing.inputs[index];
}

auto Data::getTrainingLabel(const int index) const -> int { return this->problemComposite->getTrainingLabel(index); }

auto Data::getTestingLabel(const int index) const -> int { return this->problemComposite->getTestingLabel(index); }

auto Data::getTrainingOutputs(const int index, const int batchSize) -> const std::vector<float>&
{
    return this->problemComposite->getTrainingOutputs(index, batchSize);
}

auto Data::getTestingOutputs(const int index) const -> const std::vector<float>&
{
    return this->problemComposite->getTestingOutputs(index);
}

void Data::setPrecision(const float value)
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

auto Data::getPrecision() const -> float
{
    if (this->typeOfProblem == problem::regression)
    {
        return this->precision;
    }
    throw std::runtime_error("Precision is only for regression problems.");
}

void Data::setSeparator(const float value)
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

auto Data::getSeparator() const -> float
{
    if (this->typeOfProblem == problem::classification || this->typeOfProblem == problem::multipleClassification)
    {
        return this->separator;
    }
    throw std::runtime_error("Separator is only for classification and multiple classification problems.");
}
}  // namespace snn
