#include "CompositeForTemporalData.hpp"

#include <algorithm>

namespace snn::internal
{
CompositeForTemporalData::CompositeForTemporalData(Data* data)
    : TemporalComposite(data)
{
    for (int i = 0; i > this->data->training.size; ++i)
    {
        if (this->data->training.areFirstDataOfTemporalSequence[i])
        {
            this->indexesForShuffles.push_back(i);
        }
    }
}

void CompositeForTemporalData::shuffle()
{
    std::ranges::shuffle(this->indexesForShuffles, tools::Rng());

    for (size_t i = 0, j = 0; i < this->indexesForShuffles.size(); ++i)
    {
        this->data->training.shuffledIndexes[j++] = this->indexesForShuffles[i];

        int index = this->indexesForShuffles[i] + 1;
        while (!this->data->training.areFirstDataOfTemporalSequence[index])
        {
            this->data->training.shuffledIndexes[j++] = index++;
        }
    }
}

void CompositeForTemporalData::unshuffle() { this->TemporalComposite::unshuffle(); }

auto CompositeForTemporalData::isFirstTrainingDataOfTemporalSequence(int index) const -> bool
{
    return this->data->training.areFirstDataOfTemporalSequence[index];
}

auto CompositeForTemporalData::isFirstTestingDataOfTemporalSequence(int index) const -> bool
{
    return this->data->testing.areFirstDataOfTemporalSequence[index];
}

auto CompositeForTemporalData::needToTrainOnTrainingData([[maybe_unused]] int index) const -> bool { return true; }

auto CompositeForTemporalData::needToEvaluateOnTestingData(int index) const -> bool
{
    return this->data->testing.needToEvaluateOnData[index];
}

auto CompositeForTemporalData::isValid() const -> errorType
{
    if (static_cast<int>(this->data->training.areFirstDataOfTemporalSequence.size()) != this->data->training.size ||
        static_cast<int>(this->data->testing.areFirstDataOfTemporalSequence.size()) != this->data->testing.size ||
        !this->data->training.needToTrainOnData.empty() || !this->data->testing.needToTrainOnData.empty() ||
        !this->data->training.needToEvaluateOnData.empty() ||
        static_cast<int>(this->data->testing.needToEvaluateOnData.size()) != this->data->testing.size)
    {
        return errorType::compositeForTemporalDataEmpty;
    }

    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
