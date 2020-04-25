#include "CompositeForTemporalData.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForTemporalData::CompositeForTemporalData(Set set[2])
    : TemporalComposite(set)
{
}

void CompositeForTemporalData::shuffle()
{
    throw NotImplementedException();
}

void CompositeForTemporalData::unshuffle()
{
    throw NotImplementedException();
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

int CompositeForTemporalData::isValid()
{
    if (!this->sets[testing].needToLearnData.empty())
        return 404;
    return this->TemporalComposite::isValid();
}
