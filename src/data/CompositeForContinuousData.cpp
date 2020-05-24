#include <algorithm>
#include <random>
#include "CompositeForContinuousData.hpp"
#include "../tools/ExtendedExpection.hpp"
#include "../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForContinuousData::CompositeForContinuousData(Set sets[2], int numberOfRecurrences)
    : TemporalComposite(sets), numberOfRecurrences(numberOfRecurrences)
{
    if (this->numberOfRecurrences < 1)
        throw runtime_error("The number of recurrence must be > 1 for continuous data");

    this->sets[training].numberOfTemporalSequence = 1;
    this->sets[testing].numberOfTemporalSequence = 1;

    this->divide = this->sets[training].size / this->numberOfRecurrences;
    this->rest = this->sets[training].size / this->numberOfRecurrences;

    this->indexesForShuffles.resize(this->divide);
    for(int i = 0; i < this->divide; ++i)
    {
        this->indexesForShuffles[i] = i;
    }
    this->sets[training].needToTrainOnData  = vector(this->sets[training].size, true);
    this->sets[training].areFirstDataOfTemporalSequence = vector(this->sets[training].size, false);
    this->sets[training].areFirstDataOfTemporalSequence[0] = true;
}

void CompositeForContinuousData::shuffle()
{
    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->indexesForShuffles.begin(), this->indexesForShuffles.end(), g);

    const int offset = Tools::randomBetween(0, this->numberOfRecurrences);
    const int lastRecurrence = offset > this->rest ? 1 : 0;

    for(int i = 0; i < this->indexesForShuffles.size() - lastRecurrence; ++i)
    {
        for(int j = 0; j < this->numberOfRecurrences; ++j)
        {
            const int index = i * this->numberOfRecurrences + j + offset;
            this->sets[training].indexesToShuffle[index] = this->indexesForShuffles[i] + j;

            if(j == 0)
                this->sets[training].areFirstDataOfTemporalSequence[j] = true;
            if(j == this->numberOfRecurrences-1)
                this->sets[training].needToTrainOnData[i] = true;
            else
                this->sets[training].needToTrainOnData[i] = false;
        }
    }
}

void CompositeForContinuousData::unshuffle()
{
    this->TemporalComposite::unshuffle();
    this->sets[training].needToTrainOnData  = vector(this->sets[training].size, true);
    this->sets[training].areFirstDataOfTemporalSequence = vector(this->sets[training].size, false);
    this->sets[training].areFirstDataOfTemporalSequence[0] = true;
}

bool CompositeForContinuousData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return this->sets[training].areFirstDataOfTemporalSequence[index];
}

bool CompositeForContinuousData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return index == 0 ? true : false;
}

bool CompositeForContinuousData::needToTrainOnTrainingData(int index) const
{
    return this->sets[training].needToTrainOnData[index];
}

bool CompositeForContinuousData::needToEvaluateOnTestingData(int index) const
{
    return true;
}

int CompositeForContinuousData::isValid()
{
    if (!this->sets[training].areFirstDataOfTemporalSequence.size() == this->sets[training].size
     || !this->sets[testing].areFirstDataOfTemporalSequence.empty()
     || !this->sets[training].needToTrainOnData.size() == this->sets[training].size
     || !this->sets[testing].needToTrainOnData.empty()
     || !this->sets[training].needToEvaluateOnData.empty()
     || !this->sets[testing].needToEvaluateOnData.empty())
        return 404;

    return this->TemporalComposite::isValid();
}
