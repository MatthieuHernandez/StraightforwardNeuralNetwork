#include "CompositeForContinuousData.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForContinuousData::CompositeForContinuousData(Set set[2])
    : TemporalComposite(set)
{
}

void CompositeForContinuousData::shuffle()
{
    throw NotImplementedException();
}

void CompositeForContinuousData::unshuffle()
{
    throw NotImplementedException();
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
    return this->sets[training].needToLearnData[index];
}

int CompositeForContinuousData::isValid()
{
    if (!this->sets[testing].areFirstDataOfTemporalSequence.empty()
     && !this->sets[testing].needToLearnData.empty())
        return 404;
    return this->TemporalComposite::isValid();
}
