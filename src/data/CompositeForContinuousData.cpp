#include <algorithm>
#include <random>
#include "CompositeForContinuousData.hpp"
#include "../tools/ExtendedExpection.hpp"
#include "../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForContinuousData::CompositeForContinuousData(Set sets[2], int numberOfRecurrence)
    : TemporalComposite(sets), numberOfRecurrence(numberOfRecurrence)
{
    this->divide = this->sets[training].size / this->numberOfRecurrence;
    this->rest = this->sets[training].size / this->numberOfRecurrence;

    this->indexesForShuffles.resize(this->divide);
    for(int i = 0; i < this->divide; ++i)
    {
        this->indexesForShuffles[i] = i;
    }
    if(this->numberOfRecurrence < 1)
        throw runtime_error("The number of recurrence must be > 1 for continuous data");
}

void CompositeForContinuousData::shuffle()
{
    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->indexesForShuffles.begin(), this->indexesForShuffles.end(), g);

    const int offset = Tools::randomBetween(0, this->numberOfRecurrence);
    const int lastRecurrence = offset > this->rest ? 1 : 0;

    for(int i = 0; i < this->indexesForShuffles.size() - lastRecurrence; ++i)
    {
        for(int j = 0; j < this->numberOfRecurrence; ++j)
        {
            const int index = i * this->numberOfRecurrence + j + offset;
            this->sets[training].indexesToShuffle[index] = this->indexesForShuffles[i] + j;
        }
    }
}

bool CompositeForContinuousData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return this->sets[training].areFirstDataOfTemporalSequence[index];
}

bool CompositeForContinuousData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return index == 0 ? true : false;
}

bool CompositeForContinuousData::needToLearnOnTrainingData(int index) const
{
    const int i = this->sets[training].indexesToShuffle[index];
    return this->sets[training].needToLearnData[i];
}

bool CompositeForContinuousData::needToEvaluateOnTestingData(int index) const
{
    return this->sets[testing].needToLearnData[index];
}

int CompositeForContinuousData::isValid()
{
    if (!this->sets[testing].areFirstDataOfTemporalSequence.empty()
     && !this->sets[testing].needToLearnData.empty())
        return 404;
    return this->TemporalComposite::isValid();
}
