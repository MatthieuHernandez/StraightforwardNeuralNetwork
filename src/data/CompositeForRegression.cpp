#include "CompositeForRegression.hpp"

#include "ExtendedExpection.hpp"

namespace snn::internal
{
CompositeForRegression::CompositeForRegression(Data* data, int numberOfLabels)
    : ProblemComposite(data, numberOfLabels)
{
}

auto CompositeForRegression::isValid() const -> errorType { return this->ProblemComposite::isValid(); }

auto CompositeForRegression::getTestingOutputs(const int index) const -> const std::vector<float>&
{
    return this->data->testing.labels[index];
}

auto CompositeForRegression::getTrainingLabel([[maybe_unused]] const int index) -> int
{
    throw ShouldNeverBeCalledException("getTrainingLabel");
}

auto CompositeForRegression::getTestingLabel([[maybe_unused]] const int index) -> int
{
    throw ShouldNeverBeCalledException("getTestingLabel");
}
}  // namespace snn::internal
