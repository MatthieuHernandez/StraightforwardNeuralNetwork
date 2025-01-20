#include "TemporalComposite.hpp"

using namespace std;
using namespace snn;
using namespace internal;

TemporalComposite::TemporalComposite(Set sets[2]) { this->sets = sets; }

void TemporalComposite::unshuffle()
{
    this->sets[training].shuffledIndexes.resize(this->sets[training].size);
    for (int i = 0; i < static_cast<int>(this->sets[training].shuffledIndexes.size()); ++i)
        this->sets[training].shuffledIndexes[i] = i;
}

auto TemporalComposite::isValid() const -> ErrorType
{
    if (this->sets == nullptr)
    {
        return ErrorType::temporalCompositeSetNull;
    }
    return ErrorType::noError;
}
