#include "ProblemComposite.hpp"

#include <algorithm>
#include <functional>

using namespace std;
using namespace snn;
using namespace internal;

ProblemComposite::ProblemComposite(Set sets[2], const int numberOfLabels)
    : numberOfLabels(numberOfLabels)
{
    this->sets = sets;
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
    int i = this->sets[training].shuffledIndexes[index];
    if (batchSize <= 1) return this->sets[training].labels[i];

    batchedLabels.resize(numberOfLabels);

    i = this->sets[training].shuffledIndexes[index];
    const auto data0 = this->sets[training].labels[i];
    i = this->sets[training].shuffledIndexes[index + 1];
    const auto data1 = this->sets[training].labels[i];
    ranges::transform(data0, data1, batchedLabels.begin(), plus<float>());

    for (int j = index + 2; j < index + batchSize; ++j)
    {
        i = this->sets[training].shuffledIndexes[j];
        const auto data = this->sets[training].labels[i];
        ranges::transform(batchedLabels, data, batchedLabels.begin(), std::plus<float>());
    }
    ranges::transform(batchedLabels, batchedLabels.begin(),
                      bind(divides<float>(), placeholders::_1, static_cast<float>(batchSize)));
    return batchedLabels;
}
