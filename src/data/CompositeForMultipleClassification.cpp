#include "CompositeForMultipleClassification.hpp"

#include "ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForMultipleClassification::CompositeForMultipleClassification(Set sets[2], int numberOfLabels)
    : ProblemComposite(sets, numberOfLabels)
{
}

auto CompositeForMultipleClassification::isValid() const -> ErrorType { return this->ProblemComposite::isValid(); }

auto CompositeForMultipleClassification::getTestingOutputs(const int index) const -> const std::vector<float>&
{
    return this->sets[testing].labels[index];
}

auto CompositeForMultipleClassification::getTrainingLabel([[maybe_unused]] const int index) -> int
{
    throw ShouldNeverBeCalledException("getTrainingLabel");
}

auto CompositeForMultipleClassification::getTestingLabel([[maybe_unused]] const int index) -> int
{
    throw ShouldNeverBeCalledException("getTestingLabel");
}
