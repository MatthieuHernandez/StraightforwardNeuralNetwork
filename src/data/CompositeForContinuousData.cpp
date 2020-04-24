#include "CompositeForContinuousData.hpp"

snn::internal::CompositeForContinuousData::CompositeForContinuousData(Set set[2])
    : TemporalComposite(set)
{
}

void snn::internal::CompositeForContinuousData::shuffle()
{
}

void snn::internal::CompositeForContinuousData::unshuffle()
{
}

int snn::internal::CompositeForContinuousData::isValid()
{
    return this->TemporalComposite::isValid();
}
