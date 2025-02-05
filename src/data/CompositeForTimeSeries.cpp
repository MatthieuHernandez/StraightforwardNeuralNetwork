#include "CompositeForTimeSeries.hpp"

#include <algorithm>

#include "Tools.hpp"

namespace snn::internal
{
CompositeForTimeSeries::CompositeForTimeSeries(Data* data, int numberOfRecurrences)
    : TemporalComposite(data),
      numberOfRecurrences(numberOfRecurrences)
{
    if (this->numberOfRecurrences < 1)
    {
        throw std::runtime_error("The number of recurrence must be >= 1 for time series.");
    }

    this->data->training.numberOfTemporalSequence = 1;
    this->data->testing.numberOfTemporalSequence = 1;

    this->divide = this->data->training.size / (this->numberOfRecurrences + 1);
    this->rest = this->data->training.size % (this->numberOfRecurrences + 1);

    this->indexesForShuffling.resize(this->divide);
    for (int i = 0; i < this->divide; ++i)
    {
        this->indexesForShuffling[i] = i;
    }
    this->data->training.needToTrainOnData = std::vector(this->data->training.size, true);
    this->data->training.areFirstDataOfTemporalSequence = std::vector(this->data->training.size, false);
    this->data->training.areFirstDataOfTemporalSequence[0] = true;
    this->offset = 0;
}

void CompositeForTimeSeries::shuffle()
{
    std::ranges::shuffle(this->indexesForShuffling, tools::rng);
    for (auto i = this->data->training.size - (this->numberOfRecurrences + 1); i < this->data->training.size; ++i)
    {
        this->data->training.needToTrainOnData[i] = false;
    }
    for (int i = 0; i < offset; ++i)
    {
        this->data->training.shuffledIndexes[i] = i;
        this->data->training.needToTrainOnData[i] = true;
        this->data->training.areFirstDataOfTemporalSequence[i] = false;
    }
    this->data->training.areFirstDataOfTemporalSequence[0] = true;
    int iForIndex = 0;
    for (const auto shuffledIndex : this->indexesForShuffling)
    {
        const int maxIndex = (shuffledIndex * (this->numberOfRecurrences + 1)) + this->numberOfRecurrences + offset;
        if (maxIndex < this->data->training.size)
        {
            for (int r = 0; r < this->numberOfRecurrences + 1; ++r)
            {
                const int index = (iForIndex * (this->numberOfRecurrences + 1)) + r + offset;
                this->data->training.shuffledIndexes[index] =
                    shuffledIndex * (this->numberOfRecurrences + 1) + r + offset;

                if (r != 0)
                {
                    this->data->training.areFirstDataOfTemporalSequence[index] = false;
                }
                else
                {
                    this->data->training.areFirstDataOfTemporalSequence[index] = true;
                }

                if (r == this->numberOfRecurrences && index >= offset)
                {
                    this->data->training.needToTrainOnData[index] = true;
                }
                else
                {
                    this->data->training.needToTrainOnData[index] = false;
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
    this->data->training.needToTrainOnData = std::vector(this->data->training.size, true);
    this->data->training.areFirstDataOfTemporalSequence = std::vector(this->data->training.size, false);
    this->data->training.areFirstDataOfTemporalSequence[0] = true;
}

auto CompositeForTimeSeries::isFirstTrainingDataOfTemporalSequence(int index) const -> bool
{
    return this->data->training.areFirstDataOfTemporalSequence[index];
}

auto CompositeForTimeSeries::isFirstTestingDataOfTemporalSequence(int index) const -> bool { return index == 0; }

auto CompositeForTimeSeries::needToTrainOnTrainingData(int index) const -> bool
{
    return this->data->training.needToTrainOnData[index];
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
    if (static_cast<int>(this->data->training.areFirstDataOfTemporalSequence.size()) != this->data->training.size ||
        !this->data->testing.areFirstDataOfTemporalSequence.empty() ||
        static_cast<int>(this->data->training.needToTrainOnData.size()) != this->data->training.size ||
        !this->data->testing.needToTrainOnData.empty() || !this->data->training.needToEvaluateOnData.empty() ||
        !this->data->testing.needToEvaluateOnData.empty())
    {
        return errorType::compositeForTimeSeriesEmpty;
    }

    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
