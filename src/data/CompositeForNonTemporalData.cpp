#include "CompositeForNonTemporalData.hpp"

#include <algorithm>

namespace snn::internal
{
CompositeForNonTemporalData::CompositeForNonTemporalData(Dataset* set)
    : TemporalComposite(set)
{
    this->set->training.numberOfTemporalSequence = 0;
    this->set->testing.numberOfTemporalSequence = 0;
}

void CompositeForNonTemporalData::shuffle()  // TODO(matth): also need learning to shuffle
{
    std::ranges::shuffle(this->set->training.shuffledIndexes, tools::rng);
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
    if (!this->set->training.areFirstDataOfTemporalSequence.empty() ||
        !this->set->testing.areFirstDataOfTemporalSequence.empty() || !this->set->training.needToTrainOnData.empty() ||
        !this->set->testing.needToTrainOnData.empty() || !this->set->training.needToEvaluateOnData.empty() ||
        !this->set->testing.needToEvaluateOnData.empty())
    {
        return errorType::compositeForNonTemporalDataEmpty;
    }
    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
