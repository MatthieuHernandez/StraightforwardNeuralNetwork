#pragma once
#include "Error.hpp"
#include "ProblemComposite.hpp"

namespace snn::internal
{
class CompositeForClassification : public ProblemComposite
{
    public:
        CompositeForClassification(Set sets[2], int numberOfLabels);

        [[nodiscard]] auto isValid() const -> ErrorType final;

        [[nodiscard]] auto getTestingOutputs(const int) const -> const std::vector<float>& final;
        [[nodiscard]] auto getTrainingLabel(const int) -> int final;
        [[nodiscard]] auto getTestingLabel(const int) -> int final;
};
}  // namespace snn::internal