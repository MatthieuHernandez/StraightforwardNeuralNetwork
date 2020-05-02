#include "TemporalComposite.hpp"

using namespace std;
using namespace snn;
using namespace internal;

TemporalComposite::TemporalComposite(Set sets[2])
{
    this->sets = sets;
}

void TemporalComposite::unshuffle()
{
    this->sets[training].indexesToShuffle.resize(this->sets[training].size);
    for (int i = 0; i < static_cast<int>(this->sets[training].indexesToShuffle.size()); ++i)
        this->sets[training].indexesToShuffle[i] = i;
}

int TemporalComposite::isValid()
{
    if (&this->sets[training] == nullptr
     && &this->sets[testing] == nullptr)
        return 402;
    return 0;
}
