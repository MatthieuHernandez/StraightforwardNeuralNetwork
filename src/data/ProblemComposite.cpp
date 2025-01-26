#include "ProblemComposite.hpp"

#include <algorithm>
#include <functional>

namespace snn::internal
{
ProblemComposite::ProblemComposite(Dataset* set, const int numberOfLabels)
    : numberOfLabels(numberOfLabels),
      set(set)
{
}

auto ProblemComposite::isValid() const -> errorType
{
    if (this->set == nullptr)
    {
        return errorType::dataSetNull;
    }
    return errorType::noError;
}

auto ProblemComposite::getTrainingOutputs(const int index, const int batchSize) -> const std::vector<float>&
{
    int idx = this->set->training.shuffledIndexes[index];
    if (batchSize <= 1)
    {
        return this->set->training.labels[idx];
    }
    batchedLabels.resize(numberOfLabels);

    idx = this->set->training.shuffledIndexes[index];
    const auto data0 = this->set->training.labels[idx];
    idx = this->set->training.shuffledIndexes[index + 1];
    const auto data1 = this->set->training.labels[idx];
    std::ranges::transform(data0, data1, batchedLabels.begin(), std::plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        idx = this->set->training.shuffledIndexes[j];
        const auto data = this->set->training.labels[idx];
        std::ranges::transform(batchedLabels, data, batchedLabels.begin(), std::plus<float>());
    }
    std::ranges::transform(batchedLabels, batchedLabels.begin(),
                           bind(std::divides<float>(), std::placeholders::_1, static_cast<float>(batchSize)));
    return batchedLabels;
}
}  // namespace snn::internal
