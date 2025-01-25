#pragma once
#include "ProblemComposite.hpp"

namespace snn::internal
{
class CompositeForMultipleClassification : public ProblemComposite
{
    public:
        CompositeForMultipleClassification(Set sets[2], int numberOfLabels);

        [[nodiscard]] auto isValid() const -> ErrorType override;

        [[nodiscard]] auto getTestingOutputs(const int) const -> const std::vector<float>& override;
        [[nodiscard]] auto getTrainingLabel(const int) -> int override;
        [[nodiscard]] auto getTestingLabel(const int) -> int override;
};
}  // namespace snn::internal
