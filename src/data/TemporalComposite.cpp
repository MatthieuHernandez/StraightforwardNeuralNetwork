#include "TemporalComposite.hpp"

namespace snn::internal
{
TemporalComposite::TemporalComposite(Dataset* set)
    : set(set)
{
}

void TemporalComposite::unshuffle()
{
    this->set->training.shuffledIndexes.resize(this->set->training.size);
    for (int i = 0; i < static_cast<int>(this->set->training.shuffledIndexes.size()); ++i)
    {
        this->set->training.shuffledIndexes[i] = i;
    }
}

auto TemporalComposite::isValid() const -> errorType
{
    if (this->set == nullptr)
    {
        return errorType::temporalCompositeSetNull;
    }
    return errorType::noError;
}
}  // namespace snn::internal
