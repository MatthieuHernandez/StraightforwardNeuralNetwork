#include "CompositeForTimeSeries.hpp"

#include <algorithm>
#include <random>

#include "ExtendedExpection.hpp"
#include "Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForTimeSeries::CompositeForTimeSeries(Set sets[2], int numberOfRecurrences)
    : TemporalComposite(sets),
      numberOfRecurrences(numberOfRecurrences)
{
    if (this->numberOfRecurrences < 1) throw runtime_error("The number of recurrence must be >= 1 for time series.");

    this->sets[training].numberOfTemporalSequence = 1;
    this->sets[testing].numberOfTemporalSequence = 1;

    this->divide = this->sets[training].size / (this->numberOfRecurrences + 1);
    this->rest = this->sets[training].size % (this->numberOfRecurrences + 1);

    this->indexesForShuffling.resize(this->divide);
    for (int i = 0; i < this->divide; ++i)
    {
        this->indexesForShuffling[i] = i;
    }
    this->sets[training].needToTrainOnData = vector(this->sets[training].size, true);
    this->sets[training].areFirstDataOfTemporalSequence = vector(this->sets[training].size, false);
    this->sets[training].areFirstDataOfTemporalSequence[0] = true;
    this->offset = 0;
}

void CompositeForTimeSeries::shuffle()
{
    ranges::shuffle(this->indexesForShuffling, tools::rng);

    for (int i = this->sets[training].size - (this->numberOfRecurrences + 1); i < this->sets[training].size; ++i)
    {
        this->sets[training].needToTrainOnData[i] = false;
    }

    for (int i = 0; i < offset; ++i)
    {
        this->sets[training].shuffledIndexes[i] = i;
        this->sets[training].needToTrainOnData[i] = true;
        this->sets[training].areFirstDataOfTemporalSequence[i] = false;
    }
    this->sets[training].areFirstDataOfTemporalSequence[0] = true;

    int iForIndex = 0;
    for (size_t i = 0; i < this->indexesForShuffling.size(); ++i)
    {
        const int maxIndex =
            this->indexesForShuffling[i] * (this->numberOfRecurrences + 1) + this->numberOfRecurrences + offset;
        if (maxIndex < this->sets[training].size)
        {
            for (int j = 0; j < this->numberOfRecurrences + 1; ++j)
            {
                const int index = iForIndex * (this->numberOfRecurrences + 1) + j + offset;
                this->sets[training].shuffledIndexes[index] =
                    this->indexesForShuffling[i] * (this->numberOfRecurrences + 1) + j + offset;

                if (j != 0)
                    this->sets[training].areFirstDataOfTemporalSequence[index] = false;
                else
                    this->sets[training].areFirstDataOfTemporalSequence[index] = true;

                if (j == this->numberOfRecurrences && index >= offset)
                    this->sets[training].needToTrainOnData[index] = true;
                else
                    this->sets[training].needToTrainOnData[index] = false;
            }
            iForIndex++;
        }
    }
    this->offset = (this->offset + 1) % (this->numberOfRecurrences + 1);
}

void CompositeForTimeSeries::unshuffle()
{
    this->TemporalComposite::unshuffle();
    this->sets[training].needToTrainOnData = vector(this->sets[training].size, true);
    this->sets[training].areFirstDataOfTemporalSequence = vector(this->sets[training].size, false);
    this->sets[training].areFirstDataOfTemporalSequence[0] = true;
}

bool CompositeForTimeSeries::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return this->sets[training].areFirstDataOfTemporalSequence[index];
}

bool CompositeForTimeSeries::isFirstTestingDataOfTemporalSequence(int index) const { return index == 0 ? true : false; }

bool CompositeForTimeSeries::needToTrainOnTrainingData(int index) const
{
    return this->sets[training].needToTrainOnData[index];
}

bool CompositeForTimeSeries::needToEvaluateOnTestingData([[maybe_unused]] int index) const
{
    // Skip firsts testing data can be distort the accuracy
    /*if(index < this->numberOfRecurrences)
        return false;*/
    return true;
}

int CompositeForTimeSeries::isValid()
{
    if ((int)this->sets[training].areFirstDataOfTemporalSequence.size() != this->sets[training].size ||
        !this->sets[testing].areFirstDataOfTemporalSequence.empty() ||
        (int)this->sets[training].needToTrainOnData.size() != this->sets[training].size ||
        !this->sets[testing].needToTrainOnData.empty() || !this->sets[training].needToEvaluateOnData.empty() ||
        !this->sets[testing].needToEvaluateOnData.empty())
        return 404;

    return this->TemporalComposite::isValid();
}
