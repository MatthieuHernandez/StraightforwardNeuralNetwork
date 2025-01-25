#include "CompositeForTemporalData.hpp"

#include <algorithm>
#include <random>

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
    ranges::shuffle(this->indexesForShuffles, tools::rng);

    for (size_t i = 0, j = 0; i < this->indexesForShuffles.size(); ++i)
    {
        this->sets[training].shuffledIndexes[j++] = this->indexesForShuffles[i];

        int index = this->indexesForShuffles[i] + 1;
        while (!this->sets[training].areFirstDataOfTemporalSequence[index])
        {
            this->sets[training].shuffledIndexes[j++] = index++;
        }
    }
}

void CompositeForTemporalData::unshuffle() { this->TemporalComposite::unshuffle(); }

auto CompositeForTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const -> bool
{
    return this->sets[training].areFirstDataOfTemporalSequence[index];
}

auto CompositeForTemporalData::isFirstTestingDataOfTemporalSequence(int index) const -> bool
{
    return this->sets[testing].areFirstDataOfTemporalSequence[index];
}

auto CompositeForTemporalData::needToTrainOnTrainingData([[maybe_unused]] int index) const -> bool { return true; }

auto CompositeForTemporalData::needToEvaluateOnTestingData(int index) const -> bool
{
    return this->sets[testing].needToEvaluateOnData[index];
}

auto CompositeForTemporalData::isValid() const -> ErrorType
{
    if ((int)this->sets[training].areFirstDataOfTemporalSequence.size() != this->sets[training].size ||
        (int)this->sets[testing].areFirstDataOfTemporalSequence.size() != this->sets[testing].size ||
        !this->sets[training].needToTrainOnData.empty() || !this->sets[testing].needToTrainOnData.empty() ||
        !this->sets[training].needToEvaluateOnData.empty() ||
        (int)this->sets[testing].needToEvaluateOnData.size() != this->sets[testing].size)
        return ErrorType::compositeForTemporalDataEmpty;

    return this->TemporalComposite::isValid();
}
