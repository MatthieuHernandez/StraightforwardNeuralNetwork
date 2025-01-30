#include "TemporalComposite.hpp"

namespace snn::internal
{
TemporalComposite::TemporalComposite(Data* data)
    : data(data)
{
}

void TemporalComposite::unshuffle()
{
    this->data->training.shuffledIndexes.resize(this->data->training.size);
    for (int i = 0; i < static_cast<int>(this->data->training.shuffledIndexes.size()); ++i)
    {
        this->data->training.shuffledIndexes[i] = i;
    }
}

auto TemporalComposite::isValid() const -> errorType
{
    if (this->data == nullptr)
    {
        return errorType::temporalCompositeSetNull;
    }
    return errorType::noError;
}
}  // namespace snn::internal
