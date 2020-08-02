#include "ProblemComposite.hpp"

using namespace std;
using namespace snn;
using namespace internal;

ProblemComposite::ProblemComposite(Set sets[2])
{
    this->sets = sets;
}

int ProblemComposite::isValid()
{
    if (&this->sets[training] == nullptr
        && &this->sets[testing] == nullptr)
        return 402;
    return 0;
}

const std::vector<float>& ProblemComposite::getTrainingOutputs(int index) const
{
    const int i = this->sets[training].shuffledIndexes[index];
    return this->sets[training].labels[i];
}
