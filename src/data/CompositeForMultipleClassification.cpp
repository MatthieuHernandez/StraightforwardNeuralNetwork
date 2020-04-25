#include "CompositeForMultipleClassification.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForMultipleClassification::CompositeForMultipleClassification(Set sets[2])
    : ProblemComposite(sets)
{
}

const std::vector<float>& CompositeForMultipleClassification::getTestingOutputs(const int index)
{
    return this->sets[testing].labels[index];
}

int CompositeForMultipleClassification::getTrainingLabel(const int index)
{
    throw ShouldNeverBeCalledException("getTrainingLabel");
}

int CompositeForMultipleClassification::getTestingLabel(const int index)
{
    throw ShouldNeverBeCalledException("getTestingLabel");
}
