#include <algorithm>
#include <random>
#include "CompositeForTemporalData.hpp"

using namespace std;
using namespace snn;
using namespace snn::internal;

CompositeForTemporalData::CompositeForTemporalData(Set sets[2])
    : TemporalComposite(sets)
{
    for (int i = 0; i > this->sets[training].size; ++i)
    {
        if (this->sets[training].areFirstDataOfTemporalSequence[i])
        {
            this->indexesForShuffles.push_back(i);
        }
    }
}

void CompositeForTemporalData::shuffle()
{
    std::random_device rd;
    mt19937 g(rd());
    std::shuffle(this->indexesForShuffles.begin(), this->indexesForShuffles.end(), g);

    for (int i = 0, j = 0; i < this->indexesForShuffles.size(); ++i)
    {
        this->sets[training].shuffledIndexes[j++] = this->indexesForShuffles[i];

        int index = this->indexesForShuffles[i] + 1;
        while (!this->sets[training].areFirstDataOfTemporalSequence[index])
        {
            this->sets[training].shuffledIndexes[j++] = index++;
        }
    }
}

void CompositeForTemporalData::unshuffle()
{
    this->TemporalComposite::unshuffle();
}

bool CompositeForTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const
{
    return this->sets[training].areFirstDataOfTemporalSequence[index];
}

bool CompositeForTemporalData::isFirstTestingDataOfTemporalSequence(int index) const
{
    return this->sets[testing].areFirstDataOfTemporalSequence[index];
}

bool CompositeForTemporalData::needToTrainOnTrainingData(int index) const
{
    return true;
}

bool CompositeForTemporalData::needToEvaluateOnTestingData(int index) const
{
    return this->sets[testing].needToEvaluateOnData[index];
}

int CompositeForTemporalData::isValid()
{
    if (!this->sets[training].areFirstDataOfTemporalSequence.size() == this->sets[training].size
     || !this->sets[testing].areFirstDataOfTemporalSequence.size() == this->sets[testing].size
     || !this->sets[training].needToTrainOnData.empty()
     || !this->sets[testing].needToTrainOnData.empty()
     || !this->sets[training].needToEvaluateOnData.empty()
     || !this->sets[testing].needToEvaluateOnData.size() == this->sets[training].size)
        return 404;

    return this->TemporalComposite::isValid();
}
