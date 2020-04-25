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
    this->sets[training].indexesToShuffle.resize(this->sets[training].size);
    for (int i = 0; i < static_cast<int>(this->sets[training].indexesToShuffle.size()); ++i)
        this->sets[training].indexesToShuffle[i] = i;
}

bool CompositeForNonTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return false;
}

bool CompositeForNonTemporalData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return false;
}

bool CompositeForNonTemporalData::needToLearnOnTrainingData(int index) const
{
    return true;
}

int CompositeForNonTemporalData::isValid()
//TODO: also need learning to shuffle always true override needlearn and is firstJE le trouve mal method
{
    if (!this->sets[training].areFirstDataOfTemporalSequence.empty()
     && !this->sets[testing].areFirstDataOfTemporalSequence.empty()
     && !this->sets[training].needToLearnData.empty()
     && !this->sets[testing].needToLearnData.empty())
        return 404;
    return this->TemporalComposite::isValid();
}
