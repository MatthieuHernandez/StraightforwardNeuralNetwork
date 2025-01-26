#include "CompositeForTemporalData.hpp"

#include <algorithm>

namespace snn::internal
{
CompositeForTemporalData::CompositeForTemporalData(Dataset* set)
    : TemporalComposite(set)
{
    for (int i = 0; i > this->set->training.size; ++i)
    {
        if (this->set->training.areFirstDataOfTemporalSequence[i])
        {
            this->indexesForShuffles.push_back(i);
        }
    }
}

void CompositeForTemporalData::shuffle()
{
    std::ranges::shuffle(this->indexesForShuffles, tools::rng);

    for (size_t i = 0, j = 0; i < this->indexesForShuffles.size(); ++i)
    {
        this->set->training.shuffledIndexes[j++] = this->indexesForShuffles[i];

        int index = this->indexesForShuffles[i] + 1;
        while (!this->set->training.areFirstDataOfTemporalSequence[index])
        {
            this->set->training.shuffledIndexes[j++] = index++;
        }
    }
}

void CompositeForTemporalData::unshuffle() { this->TemporalComposite::unshuffle(); }

auto CompositeForTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const -> bool
{
    return this->set->training.areFirstDataOfTemporalSequence[index];
}

auto CompositeForTemporalData::isFirstTestingDataOfTemporalSequence(int index) const -> bool
{
    return this->set->testing.areFirstDataOfTemporalSequence[index];
}

auto CompositeForTemporalData::needToTrainOnTrainingData([[maybe_unused]] int index) const -> bool { return true; }

auto CompositeForTemporalData::needToEvaluateOnTestingData(int index) const -> bool
{
    return this->set->testing.needToEvaluateOnData[index];
}

auto CompositeForTemporalData::isValid() const -> errorType
{
    if (static_cast<int>(this->set->training.areFirstDataOfTemporalSequence.size()) != this->set->training.size ||
        static_cast<int>(this->set->testing.areFirstDataOfTemporalSequence.size()) != this->set->testing.size ||
        !this->set->training.needToTrainOnData.empty() || !this->set->testing.needToTrainOnData.empty() ||
        !this->set->training.needToEvaluateOnData.empty() ||
        static_cast<int>(this->set->testing.needToEvaluateOnData.size()) != this->set->testing.size)
    {
        return errorType::compositeForTemporalDataEmpty;
    }

    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
