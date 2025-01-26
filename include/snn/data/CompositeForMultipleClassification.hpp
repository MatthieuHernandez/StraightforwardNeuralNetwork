#pragma once
#include "ProblemComposite.hpp"

namespace snn::internal
{
class CompositeForMultipleClassification : public ProblemComposite
{
    public:
        CompositeForMultipleClassification(Dataset* set, int numberOfLabels);

        [[nodiscard]] auto isValid() const -> errorType final;

        [[nodiscard]] auto getTestingOutputs(int index) const -> const std::vector<float>& final;
        [[nodiscard]] auto getTrainingLabel(int index) -> int final;
        [[nodiscard]] auto getTestingLabel(int index) -> int final;
};
}  // namespace snn::internal
