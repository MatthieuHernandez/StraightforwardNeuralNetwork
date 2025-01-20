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

const std::vector<float>& CompositeForMultipleClassification::getTestingOutputs(const int index) const
{
    return this->sets[testing].labels[index];
}

int CompositeForMultipleClassification::getTrainingLabel([[maybe_unused]] const int index)
{
    throw ShouldNeverBeCalledException("getTrainingLabel");
}

int CompositeForMultipleClassification::getTestingLabel([[maybe_unused]] const int index)
{
    throw ShouldNeverBeCalledException("getTestingLabel");
}
