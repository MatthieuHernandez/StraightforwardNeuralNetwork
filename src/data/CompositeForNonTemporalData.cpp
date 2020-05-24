#include <algorithm>
#include <random>
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
    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->sets[training].indexesToShuffle.begin(), this->sets[training].indexesToShuffle.end(), g);
}

void CompositeForNonTemporalData::unshuffle()
{
    this->TemporalComposite::unshuffle();
}

bool CompositeForNonTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return false;
}

bool CompositeForNonTemporalData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return false;
}

bool CompositeForNonTemporalData::needToTrainOnTrainingData(int index) const
{
    return true;
}

bool CompositeForNonTemporalData::needToEvaluateOnTestingData(int index) const
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
