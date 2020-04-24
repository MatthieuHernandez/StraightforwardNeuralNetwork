#include "CompositeForNonTemporalData.hpp"
#include <random>

using namespace std;
using namespace snn;
using namespace internal;

CompositeForNonTemporalData::CompositeForNonTemporalData(Set sets[2])
    : TemporalComposite(sets)
{
}

void CompositeForNonTemporalData::shuffle() //TODO: also need learning to shuffle
{
    if (this->sets[training].indexesToShuffle.empty())
    {
        this->sets[training].indexesToShuffle.resize(this->sets[training].size);
        for (int i = 0; i < static_cast<int>(this->sets[training].indexesToShuffle.size()); i++)
            this->sets[training].indexesToShuffle[i] = i;
    }

    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->sets[training].indexesToShuffle.begin(), this->sets[training].indexesToShuffle.end(), g);
}

void CompositeForNonTemporalData::unshuffle()
{
    sets[training].indexesToShuffle.resize(sets[training].size);
    for (int i = 0; i < static_cast<int>(sets[training].indexesToShuffle.size()); ++i)
        sets[training].indexesToShuffle[i] = i;
}

int CompositeForNonTemporalData::isValid() //TODO: also need learning to shuffle always true override needlearn and is first method
{
    return this->TemporalComposite::isValid();
}
