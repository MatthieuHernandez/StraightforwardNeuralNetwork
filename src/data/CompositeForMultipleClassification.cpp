#include "CompositeForMultipleClassification.hpp"

#include "ExtendedExpection.hpp"

namespace snn::internal
{
CompositeForMultipleClassification::CompositeForMultipleClassification(Data* data, int numberOfLabels)
    : ProblemComposite(data, numberOfLabels)
{
}

auto CompositeForMultipleClassification::isValid() const -> errorType { return this->ProblemComposite::isValid(); }

auto CompositeForMultipleClassification::getTestingOutputs(const int index) const -> const std::vector<float>&
{
    return this->data->testing.labels[index];
}

auto CompositeForMultipleClassification::getTrainingLabel([[maybe_unused]] const int index) -> int
{
    throw ShouldNeverBeCalledException("getTrainingLabel");
}

auto CompositeForMultipleClassification::getTestingLabel([[maybe_unused]] const int index) -> int
{
    throw ShouldNeverBeCalledException("getTestingLabel");
}
}  // namespace snn::internal
