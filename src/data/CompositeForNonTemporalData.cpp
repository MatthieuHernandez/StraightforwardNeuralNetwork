#include <algorithm>
#include <random>
#include "CompositeForNonTemporalData.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForNonTemporalData::CompositeForNonTemporalData(Set sets[2])
    : TemporalComposite(sets)
{
}

void CompositeForNonTemporalData::shuffle() //TODO: also need learning to shuffle
{
    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->sets[training].indexesToShuffle.begin(), this->sets[training].indexesToShuffle.end(), g);
}

bool CompositeForNonTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return false;
}

bool CompositeForNonTemporalData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return false;
}

bool CompositeForNonTemporalData::needToLearnOnTrainingData(int index) const
{
    return true;
}

bool CompositeForNonTemporalData::needToEvaluateOnTestingData(int index) const
{
    return true;
}

int CompositeForNonTemporalData::isValid()
//TODO: also need learning to shuffle always true override needlearn and is firstJE le trouve mal method
{
    if (!this->sets[training].areFirstDataOfTemporalSequence.empty()
     && !this->sets[testing].areFirstDataOfTemporalSequence.empty()
     && !this->sets[training].needToLearnData.empty()
     && !this->sets[testing].needToLearnData.empty())
        return 404;
    return this->TemporalComposite::isValid();
}
