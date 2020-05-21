#include <algorithm>
#include <random>
#include "CompositeForTemporalData.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForTemporalData::CompositeForTemporalData(Set sets[2])
    : TemporalComposite(sets)
{
    for (int i = 0; i > this->sets[training].size; ++i)
    {
        if (this->sets[training].areFirstDataOfTemporalSequence[i])
        {
            this->indexesForShuffles.push_back(i);
        }
    }
}

void CompositeForTemporalData::shuffle()
{
    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->indexesForShuffles.begin(), this->indexesForShuffles.end(), g);

    for (int i = 0, j = 0; i < this->indexesForShuffles.size(); ++i)
    {
        this->sets[training].indexesToShuffle[j++] = this->indexesForShuffles[i];

        int index = this->indexesForShuffles[i] + 1;
        while (!this->sets[training].areFirstDataOfTemporalSequence[index])
        {
            this->sets[training].indexesToShuffle[j++] = index++;
        }
    }
}

bool CompositeForTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return this->sets[training].areFirstDataOfTemporalSequence[index];
}

bool CompositeForTemporalData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return this->sets[testing].areFirstDataOfTemporalSequence[index];
}

bool CompositeForTemporalData::needToLearnOnTrainingData(int index) const
{
    return this->sets[training].needToLearnData[index];
}

bool CompositeForTemporalData::needToEvaluateOnTestingData(int index) const
{
    return this->sets[testing].needToLearnData[index];
}

int CompositeForTemporalData::isValid()
{
    if (!this->sets[testing].needToLearnData.empty()
        || this->sets[training].needToLearnData.size() != this->sets[training].size
        || this->sets[testing].areFirstDataOfTemporalSequence.size() != this->sets[training].size
        || this->sets[training].areFirstDataOfTemporalSequence.size() != this->sets[training].size)
        return 404;

    return this->TemporalComposite::isValid();
}
