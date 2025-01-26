#include "CompositeForTimeSeries.hpp"

#include <algorithm>

#include "Tools.hpp"

namespace snn::internal
{
CompositeForTimeSeries::CompositeForTimeSeries(Dataset* set, int numberOfRecurrences)
    : TemporalComposite(set),
      numberOfRecurrences(numberOfRecurrences)
{
    if (this->numberOfRecurrences < 1)
    {
        throw std::runtime_error("The number of recurrence must be >= 1 for time series.");
    }

    this->set->training.numberOfTemporalSequence = 1;
    this->set->testing.numberOfTemporalSequence = 1;

    this->divide = static_cast<int>(this->set->training.size) / (this->numberOfRecurrences + 1);
    this->rest = this->set->training.size % (this->numberOfRecurrences + 1);

    this->indexesForShuffling.resize(this->divide);
    for (int i = 0; i < this->divide; ++i)
    {
        this->indexesForShuffling[i] = i;
    }
    this->set->training.needToTrainOnData = std::vector(this->set->training.size, true);
    this->set->training.areFirstDataOfTemporalSequence = std::vector(this->set->training.size, false);
    this->set->training.areFirstDataOfTemporalSequence[0] = true;
    this->offset = 0;
}

void CompositeForTimeSeries::shuffle()
{
    std::ranges::shuffle(this->indexesForShuffling, tools::rng);

    for (auto i = this->set->training.size - (this->numberOfRecurrences + 1); i < this->set->training.size; ++i)
    {
        this->set->training.needToTrainOnData[i] = false;
    }

    for (int i = 0; i < offset; ++i)
    {
        this->set->training.shuffledIndexes[i] = i;
        this->set->training.needToTrainOnData[i] = true;
        this->set->training.areFirstDataOfTemporalSequence[i] = false;
    }
    this->set->training.areFirstDataOfTemporalSequence[0] = true;

    int iForIndex = 0;
    for (size_t i = 0; i < this->indexesForShuffling.size(); ++i)
    {
        const int maxIndex =
            this->indexesForShuffling[i] * (this->numberOfRecurrences + 1) + this->numberOfRecurrences + offset;
        if (maxIndex < this->set->training.size)
        {
            for (int j = 0; j < this->numberOfRecurrences + 1; ++j)
            {
                const int index = iForIndex * (this->numberOfRecurrences + 1) + j + offset;
                this->set->training.shuffledIndexes[index] =
                    this->indexesForShuffling[i] * (this->numberOfRecurrences + 1) + j + offset;

                if (j != 0)
                {
                    this->set->training.areFirstDataOfTemporalSequence[index] = false;
                }
                else
                {
                    this->set->training.areFirstDataOfTemporalSequence[index] = true;
                }

                if (j == this->numberOfRecurrences && index >= offset)
                {
                    this->set->training.needToTrainOnData[index] = true;
                }
                else
                {
                    this->set->training.needToTrainOnData[index] = false;
                }
            }
            iForIndex++;
        }
    }
    this->offset = (this->offset + 1) % (this->numberOfRecurrences + 1);
}

void CompositeForTimeSeries::unshuffle()
{
    this->TemporalComposite::unshuffle();
    this->set->training.needToTrainOnData = std::vector(this->set->training.size, true);
    this->set->training.areFirstDataOfTemporalSequence = std::vector(this->set->training.size, false);
    this->set->training.areFirstDataOfTemporalSequence[0] = true;
}

auto CompositeForTimeSeries::isFirstTrainingDataOfTemporalSequence(int index) const -> bool
{
    return this->set->training.areFirstDataOfTemporalSequence[index];
}

auto CompositeForTimeSeries::isFirstTestingDataOfTemporalSequence(int index) const -> bool
{
    return index == 0 ? true : false;
}

auto CompositeForTimeSeries::needToTrainOnTrainingData(int index) const -> bool
{
    return this->set->training.needToTrainOnData[index];
}

auto CompositeForTimeSeries::needToEvaluateOnTestingData([[maybe_unused]] int index) const -> bool
{
    // Skip firsts testing data can be distort the accuracy
    /*if(index < this->numberOfRecurrences)
        return false;*/
    return true;
}

auto CompositeForTimeSeries::isValid() const -> errorType
{
    if (static_cast<int>(this->sets[training].areFirstDataOfTemporalSequence.size()) != this->sets[training].size ||
        !this->set->testing.areFirstDataOfTemporalSequence.empty() ||
        static_cast<int>(this->sets[training].needToTrainOnData.size()) != this->sets[training].size ||
        !this->set->testing.needToTrainOnData.empty() || !this->set->training.needToEvaluateOnData.empty() ||
        !this->set->testing.needToEvaluateOnData.empty())
    {
        return errorType::compositeForTimeSeriesEmpty;
    }

    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
