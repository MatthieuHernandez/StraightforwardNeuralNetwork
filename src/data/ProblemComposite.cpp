#include "ProblemComposite.hpp"

#include <functional>

namespace snn::internal
{
ProblemComposite::ProblemComposite(Set sets[2], const int numberOfLabels)
    : numberOfLabels(numberOfLabels),
      sets(sets)
{
}

auto ProblemComposite::isValid() const -> ErrorType
{
    if (this->sets == nullptr)
    {
        return ErrorType::dataSetNull;
    }
    return ErrorType::noError;
}

auto ProblemComposite::getTrainingOutputs(const int index, const int batchSize) -> const std::vector<float>&
{
    int idx = this->sets[training].shuffledIndexes[index];
    if (batchSize <= 1)
    {
        return this->sets[training].labels[idx];
    }
    batchedLabels.resize(numberOfLabels);

    idx = this->sets[training].shuffledIndexes[index];
    const auto data0 = this->sets[training].labels[idx];
    idx = this->sets[training].shuffledIndexes[index + 1];
    const auto data1 = this->sets[training].labels[idx];
    std::ranges::transform(data0, data1, batchedLabels.begin(), std::plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        idx = this->sets[training].shuffledIndexes[j];
        const auto data = this->sets[training].labels[idx];
        std::ranges::transform(batchedLabels, data, batchedLabels.begin(), std::plus<float>());
    }
    std::ranges::transform(batchedLabels, batchedLabels.begin(),
                           bind(std::divides<float>(), std::placeholders::_1, static_cast<float>(batchSize)));
    return batchedLabels;
}
}  // namespace snn::internal
