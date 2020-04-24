#include "CompositeForTemporalData.hpp"

snn::internal::CompositeForTemporalData::CompositeForTemporalData(Set set[2])
    : TemporalComposite(set)
{
}

void snn::internal::CompositeForTemporalData::shuffle()
{
}

void snn::internal::CompositeForTemporalData::unshuffle()
{
}

int snn::internal::CompositeForTemporalData::isValid()
{
    return this->TemporalComposite::isValid();
}
