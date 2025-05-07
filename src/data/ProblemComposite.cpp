#include "ProblemComposite.hpp"

#include <algorithm>
#include <functional>

namespace snn::internal
{
ProblemComposite::ProblemComposite(Data* data, const int numberOfLabels)
    : numberOfLabels(numberOfLabels),
      data(data)
{
}

auto ProblemComposite::isValid() const -> errorType
{
    if (this->data == nullptr)
    {
        return errorType::dataSetNull;
    }
    return errorType::noError;
}

auto ProblemComposite::getTrainingOutputs(const int index, const int batchSize) -> const std::vector<float>&
{
    int idx = this->data->training.shuffledIndexes[index];
    if (batchSize <= 1)
    {
        return this->data->training.labels[idx];
    }
    batchedLabels.resize(numberOfLabels);

    idx = this->data->training.shuffledIndexes[index];
    const auto data0 = this->data->training.labels[idx];
    idx = this->data->training.shuffledIndexes[index + 1];
    const auto data1 = this->data->training.labels[idx];
    std::ranges::transform(data0, data1, batchedLabels.begin(), std::plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        idx = this->data->training.shuffledIndexes[j];
        const auto dataToAdd = this->data->training.labels[idx];
        std::ranges::transform(batchedLabels, dataToAdd, batchedLabels.begin(), std::plus<float>());
    }
    std::ranges::transform(batchedLabels, batchedLabels.begin(),
                           [batchSize](auto&& value) { return value / static_cast<float>(batchSize); });
    return batchedLabels;
}
}  // namespace snn::internal
