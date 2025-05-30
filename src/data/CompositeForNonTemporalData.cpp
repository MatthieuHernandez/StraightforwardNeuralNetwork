#include "CompositeForNonTemporalData.hpp"

#include <algorithm>

namespace snn::internal
{
CompositeForNonTemporalData::CompositeForNonTemporalData(Data* data)
    : TemporalComposite(data)
{
    this->data->training.numberOfTemporalSequence = 0;
    this->data->testing.numberOfTemporalSequence = 0;
}

void CompositeForNonTemporalData::shuffle()  // TODO(matth): also need learning to shuffle
{
    std::ranges::shuffle(this->data->training.shuffledIndexes, tools::Rng());
}

void CompositeForNonTemporalData::unshuffle() { this->TemporalComposite::unshuffle(); }

auto CompositeForNonTemporalData::isFirstTrainingDataOfTemporalSequence([[maybe_unused]] int index) const -> bool
{
    return true;
}

auto CompositeForNonTemporalData::isFirstTestingDataOfTemporalSequence([[maybe_unused]] int index) const -> bool
{
    return false;
}

auto CompositeForNonTemporalData::needToTrainOnTrainingData([[maybe_unused]] int index) const -> bool { return true; }

auto CompositeForNonTemporalData::needToEvaluateOnTestingData([[maybe_unused]] int index) const -> bool { return true; }

auto CompositeForNonTemporalData::isValid() const -> errorType
{
    if (!this->data->training.areFirstDataOfTemporalSequence.empty() ||
        !this->data->testing.areFirstDataOfTemporalSequence.empty() ||
        !this->data->training.needToTrainOnData.empty() || !this->data->testing.needToTrainOnData.empty() ||
        !this->data->training.needToEvaluateOnData.empty() || !this->data->testing.needToEvaluateOnData.empty())
    {
        return errorType::compositeForNonTemporalDataEmpty;
    }
    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
