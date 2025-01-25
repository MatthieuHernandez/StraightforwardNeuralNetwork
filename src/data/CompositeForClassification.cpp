#include "CompositeForClassification.hpp"

#include "ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForClassification::CompositeForClassification(Set sets[2], int numberOfLabels)
    : ProblemComposite(sets, numberOfLabels)
{
}

auto CompositeForClassification::isValid() const -> ErrorType
{
    if (this->sets[training].labels[0].size() < 2)
    {
        return ErrorType::dataWrongLabelSize;
    };
    return this->ProblemComposite::isValid();
}

auto CompositeForClassification::getTrainingLabel(const int index) -> int
{
    for (int i = 0; i < (int)this->sets[training].labels[index].size(); i++)
    {
        if (this->sets[training].labels[index][i] == 1) return i;
    }
    throw std::runtime_error("wrong label");
}

auto CompositeForClassification::getTestingLabel(const int index) -> int
{
    for (int i = 0; i < (int)this->sets[testing].labels[index].size(); i++)
    {
        if (this->sets[testing].labels[index][i] == 1) return i;
    }
    throw std::runtime_error("wrong label");
}

auto CompositeForClassification::getTestingOutputs(const int) const -> const std::vector<float>&
{
    throw ShouldNeverBeCalledException("getTestingOutputs");
}
