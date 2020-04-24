#include "ProblemComposite.hpp"

using namespace std;
using namespace snn;
using namespace internal;

ProblemComposite::ProblemComposite(Set set[2])
{
    *this->sets = *set;
}

int ProblemComposite::isValid()
{
    if (&this->sets[training] == nullptr
        && &this->sets[testing] == nullptr)
        return 402;
    return 0;
}
