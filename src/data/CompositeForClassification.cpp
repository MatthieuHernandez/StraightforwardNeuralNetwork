#include "CompositeForClassification.hpp"

#include "ExtendedExpection.hpp"

namespace snn::internal
{
CompositeForClassification::CompositeForClassification(Data* data, int numberOfLabels)
    : ProblemComposite(data, numberOfLabels)
{
}

auto CompositeForClassification::isValid() const -> errorType
{
    if (this->data->training.labels[0].size() < 2)
    {
        return errorType::dataWrongLabelSize;
    };
    return this->ProblemComposite::isValid();
}

auto CompositeForClassification::getTrainingLabel(const int index) -> int
{
    for (int i = 0; i < static_cast<int>(this->data->training.labels[index].size()); i++)
    {
        if (this->data->training.labels[index][i] == 1)
        {
            return i;
        }
    }
    throw std::runtime_error("wrong label");
}

auto CompositeForClassification::getTestingLabel(const int index) -> int
{
    for (int i = 0; i < static_cast<int>(this->data->testing.labels[index].size()); i++)
    {
        if (this->data->testing.labels[index][i] == 1)
        {
            return i;
        }
    }
    throw std::runtime_error("wrong label");
}

auto CompositeForClassification::getTestingOutputs([[maybe_unused]] const int index) const -> const std::vector<float>&
{
    throw ShouldNeverBeCalledException("getTestingOutputs");
}
}  // namespace snn::internal
