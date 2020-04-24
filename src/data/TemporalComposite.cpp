#include "TemporalComposite.hpp"

using namespace std;
using namespace snn;
using namespace internal;

TemporalComposite::TemporalComposite(Set set[2])
{
    *this->sets = *set;
}

int TemporalComposite::isValid()
{
    if(&this->sets[training] == nullptr
    && &this->sets[testing] == nullptr)
        return 402;
    return 0;
}
