#include "CompositeForClassification.hpp"
#include "../tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

CompositeForClassification::CompositeForClassification(Set sets[2])
    : ProblemComposite(sets)
{
}

int CompositeForClassification::isValid()
{
    return this->ProblemComposite::isValid();
}

int CompositeForClassification::getTrainingLabel(const int index)
{
    for (int i = 0; i < this->sets[training].labels[index].size(); i++)
    {
        if (this->sets[training].labels[index][i] == 1)
            return i;
    }
    throw std::runtime_error("wrong label");
}

int CompositeForClassification::getTestingLabel(const int index)
{
    for (int i = 0; i < this->sets[testing].labels[index].size(); i++)
    {
        if (this->sets[testing].labels[index][i] == 1)
            return i;
    }
    throw std::runtime_error("wrong label");
}


const std::vector<float>& CompositeForClassification::getTestingOutputs(const int) const
{
    throw ShouldNeverBeCalledException("getTestingOutputs");
}
