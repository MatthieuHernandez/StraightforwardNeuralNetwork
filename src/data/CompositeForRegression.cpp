#include "CompositeForRegression.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForRegression::CompositeForRegression(Set sets[2])
    : ProblemComposite(sets)
{
}

int CompositeForRegression::isValid()
{
    return this->ProblemComposite::isValid();
}

const std::vector<float>& CompositeForRegression::getTestingOutputs(const int index)
{
    return this->sets[testing].labels[index];
}

int CompositeForRegression::getTrainingLabel(const int index)
{
    throw ShouldNeverBeCalledException("getTrainingLabel");
}

int CompositeForRegression::getTestingLabel(const int index)
{
    throw ShouldNeverBeCalledException("getTestingLabel");
}
