#include "CompositeForNonTemporalData.hpp"

namespace snn::internal
{
CompositeForNonTemporalData::CompositeForNonTemporalData(Set sets[2])
    : TemporalComposite(sets)
{
    this->sets[training].numberOfTemporalSequence = 0;
    this->sets[testing].numberOfTemporalSequence = 0;
}

void CompositeForNonTemporalData::shuffle()  // TODO: also need learning to shuffle
{
    std::ranges::shuffle(this->sets[training].shuffledIndexes, tools::rng);
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

auto CompositeForNonTemporalData::isValid() const -> ErrorType
{
    if (!this->sets[training].areFirstDataOfTemporalSequence.empty() ||
        !this->sets[testing].areFirstDataOfTemporalSequence.empty() ||
        !this->sets[training].needToTrainOnData.empty() || !this->sets[testing].needToTrainOnData.empty() ||
        !this->sets[training].needToEvaluateOnData.empty() || !this->sets[testing].needToEvaluateOnData.empty())
    {
        return ErrorType::compositeForNonTemporalDataEmpty;
    }
    return this->TemporalComposite::isValid();
}
}  // namespace snn::internal
