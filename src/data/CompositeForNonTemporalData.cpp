#include <algorithm>
#include <random>
#include <ranges>
#include "CompositeForNonTemporalData.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForNonTemporalData::CompositeForNonTemporalData(Set sets[2])
    : TemporalComposite(sets)
{
    this->sets[training].numberOfTemporalSequence = 0;
    this->sets[testing].numberOfTemporalSequence = 0;
}

void CompositeForNonTemporalData::shuffle() //TODO: also need learning to shuffle
{
    random_device rd;
    mt19937 g(rd());
    ranges::shuffle(this->sets[training].shuffledIndexes, g);
}

void CompositeForNonTemporalData::unshuffle()
{
    this->TemporalComposite::unshuffle();
}

bool CompositeForNonTemporalData::isFirstTrainingDataOfTemporalSequence([[maybe_unused]] int index) const
{
    return true;
}

bool CompositeForNonTemporalData::isFirstTestingDataOfTemporalSequence([[maybe_unused]] int index) const
{
    return false;
}

bool CompositeForNonTemporalData::needToTrainOnTrainingData([[maybe_unused]] int index) const
{
    return true;
}

bool CompositeForNonTemporalData::needToEvaluateOnTestingData([[maybe_unused]] int index) const
{
    return true;
}

int CompositeForNonTemporalData::isValid()
{
    if (!this->sets[training].areFirstDataOfTemporalSequence.empty()
     || !this->sets[testing].areFirstDataOfTemporalSequence.empty()
     || !this->sets[training].needToTrainOnData.empty()
     || !this->sets[testing].needToTrainOnData.empty()
     || !this->sets[training].needToEvaluateOnData.empty()
     || !this->sets[testing].needToEvaluateOnData.empty())
        return 404;
    return this->TemporalComposite::isValid();
}
